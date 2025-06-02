"""
Final Clean Matcher - Complete with Full Explainability
Enhanced version with detailed explanations for every match decision
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
import logging
from pathlib import Path
from collections import Counter
import math
import chardet
import spacy
import json
import re
from dataclasses import dataclass

@dataclass
class MatchExplanation:
    """Detailed explanation of why a requirement matched an activity."""
    requirement_id: str
    requirement_text: str
    activity_name: str
    combined_score: float
    
    # Score components
    semantic_score: float
    bm25_score: float
    syntactic_score: float
    domain_score: float
    query_expansion_score: float
    
    # Detailed explanations
    semantic_explanation: str
    bm25_explanation: str
    syntactic_explanation: str
    domain_explanation: str
    query_expansion_explanation: str
    
    # Key evidence
    shared_terms: List[str]
    semantic_similarity_level: str
    match_quality: str

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FinalCleanMatcher:
    """
    Final clean matcher with comprehensive explainability to reach F1@5 = 0.213 target.
    """
    
    def __init__(self, model_name: str = "en_core_web_trf"):
        """Initialize with spaCy transformer model."""
        try:
            self.nlp = spacy.load(model_name)
            logger.info(f"Loaded spaCy model: {model_name}")
        except OSError:
            logger.error(f"Could not load model {model_name}")
            raise
        
        # Debug counters (reset for each run)
        self.debug_bm25_count = 0
        self.debug_query_count = 0
        
        # Load synonym dictionary for query expansion
        try:
            with open("synonyms.json", 'r') as f:
                self.synonyms = json.load(f)
                logger.info(f"Loaded synonym dictionary with {len(self.synonyms)} entries")
        except FileNotFoundError:
            self.synonyms = {}
            logger.warning("No synonym dictionary found")
    
    def _detect_encoding(self, file_path: str) -> str:
        """Detect file encoding with fallback."""
        try:
            with open(file_path, 'rb') as f:
                raw_data = f.read(10000)
                result = chardet.detect(raw_data)
                return result['encoding'] or 'utf-8'
        except Exception:
            return 'utf-8'
    
    def _safe_read_csv(self, file_path: str, **kwargs) -> pd.DataFrame:
        """Safely read CSV with automatic encoding detection."""
        encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
        
        # Try detected encoding first
        detected = self._detect_encoding(file_path)
        if detected not in encodings:
            encodings.insert(0, detected)
        
        for encoding in encodings:
            try:
                df = pd.read_csv(file_path, encoding=encoding, **kwargs)
                logger.info(f"Read {file_path} with {encoding}")
                return df
            except UnicodeDecodeError:
                continue
        
        raise RuntimeError(f"Could not read {file_path} with any encoding")
    
    def extract_syntactic_features(self, doc: spacy.tokens.Doc) -> Dict[str, List]:
        """Extract syntactic features from document for structural similarity."""
        features = {
            'dep_patterns': [],
            'pos_sequence': [],
            'entity_types': [],
            'verb_frames': [],
        }
        
        for token in doc:
            if not token.is_stop and token.is_alpha:
                pattern = f"{token.dep_}:{token.pos_}"
                features['dep_patterns'].append(pattern)
                
                if token.pos_ == "VERB":
                    children = [child.dep_ for child in token.children]
                    if children:
                        features['verb_frames'].append(f"{token.lemma_}:{':'.join(sorted(children))}")
        
        pos_tags = [token.pos_ for token in doc if not token.is_space]
        for i in range(len(pos_tags) - 1):
            features['pos_sequence'].append(f"{pos_tags[i]}_{pos_tags[i+1]}")
        
        features['entity_types'] = [ent.label_ for ent in doc.ents]
        
        return features
    
    def compute_syntactic_similarity_with_explanation(self, features1: Dict, features2: Dict) -> Tuple[float, str]:
        """Compute syntactic similarity with detailed explanation."""
        total_sim = 0.0
        weights = {
            'dep_patterns': 0.4,
            'pos_sequence': 0.3, 
            'entity_types': 0.2,
            'verb_frames': 0.1
        }
        
        explanations = []
        
        for feature_type, weight in weights.items():
            set1 = set(features1.get(feature_type, []))
            set2 = set(features2.get(feature_type, []))
            
            if not set1 and not set2:
                jaccard = 1.0
                explanations.append(f"{feature_type}: both empty (1.0)")
            elif not set1 or not set2:
                jaccard = 0.0
                explanations.append(f"{feature_type}: one empty (0.0)")
            else:
                intersection = len(set1 & set2)
                union = len(set1 | set2)
                jaccard = intersection / union if union > 0 else 0.0
                shared = list(set1 & set2)[:2]  # Show top 2 shared features
                explanations.append(f"{feature_type}: {intersection}/{union}={jaccard:.2f}")
                if shared and len(shared) > 0:
                    explanations[-1] += f" (shared: {shared})"
            
            total_sim += weight * jaccard
        
        explanation = "; ".join(explanations[:2])  # Show top 2 components
        if len(explanations) > 2:
            explanation += f" +{len(explanations)-2} more"
            
        return total_sim, explanation
    
    def extract_domain_terms(self, corpus: List[str]) -> Dict[str, float]:
        """Extract domain-specific terms with improved filtering - auto-detects technical terms."""
        all_text = ' '.join(corpus)
        doc = self.nlp(all_text)
        
        candidates = []
        
        # Add individual technical terms (focus on meaningful nouns)
        for token in doc:
            if (token.pos_ in ['NOUN', 'PROPN'] and 
                not token.is_stop and 
                len(token.text) > 3 and
                token.text.lower() not in {'system', 'data', 'process', 'operation', 'component'}):
                candidates.append(token.lemma_.lower())
        
        # Add noun phrases but filter out generic ones
        for chunk in doc.noun_chunks:
            if (len(chunk.text.split()) <= 3 and
                'system' not in chunk.text.lower() and
                'data' not in chunk.text.lower() and
                len(chunk.text) > 5):
                candidates.append(chunk.text.lower().strip())
        
        term_freq = Counter(candidates)
        total_terms = sum(term_freq.values())
        
        domain_weights = {}
        for term, freq in term_freq.items():
            if freq >= 2:
                tech_score = 1.0
                
                # Auto-detect technical characteristics (domain-agnostic)
                # 1. Terms with numbers/special chars (technical specifications)
                if any(c.isdigit() for c in term):
                    tech_score *= 1.5
                if any(c in term for c in ['_', '-', '/']):
                    tech_score *= 1.3
                
                # 2. Longer terms (often more specific/technical)
                if len(term) > 8:
                    tech_score *= 1.2
                elif len(term) > 12:
                    tech_score *= 1.4
                
                # 3. Capitalized terms (acronyms, proper nouns)
                if term.isupper() and len(term) > 2:
                    tech_score *= 1.3
                
                # 4. Multi-word technical phrases
                if len(term.split()) > 1:
                    tech_score *= 1.2
                
                # 5. Penalize overly common terms (appears in >10% of corpus)
                if freq > total_terms * 0.1:
                    tech_score *= 0.5
                
                # 6. Boost moderately rare terms (sweet spot for technical terms)
                if 0.01 <= freq/total_terms <= 0.05:  # 1-5% frequency
                    tech_score *= 1.3
                
                # Combine frequency and technical weighting
                length_bonus = min(len(term.split()), 3) / 3.0
                freq_weight = freq / total_terms
                domain_weights[term] = freq_weight * tech_score * (1 + length_bonus)
        
        max_weight = max(domain_weights.values()) if domain_weights else 1.0
        normalized_weights = {term: weight / max_weight for term, weight in domain_weights.items()}
        
        return normalized_weights
    
    def expand_query_with_explanation(self, query_doc: spacy.tokens.Doc) -> Tuple[List[str], str]:
        """Query expansion with detailed explanation."""
        expanded_terms = []
        explanations = []
        debug_tokens = []
        
        for token in query_doc:
            if (token.pos_ in ['NOUN', 'VERB', 'ADJ'] and 
                not token.is_stop and
                len(token.text) > 2):
                
                # Debug: show what we're looking up
                lemma_key = token.lemma_.lower()
                text_key = token.text.lower()
                debug_tokens.append(f"{token.text}->{lemma_key}")
                
                # Try multiple lookup strategies
                found_synonyms = []
                
                # 1. Try lemma lookup
                if lemma_key in self.synonyms:
                    found_synonyms.extend(self.synonyms[lemma_key][:3])
                
                # 2. Try original text lookup
                if text_key in self.synonyms and text_key != lemma_key:
                    found_synonyms.extend(self.synonyms[text_key][:3])
                
                if found_synonyms:
                    expanded_terms.extend(found_synonyms)
                    explanations.append(f"'{token.text}'→{found_synonyms[:2]}")
        
        final_terms = list(set(expanded_terms))
        
        # Create explanation
        if explanations:
            explanation = "; ".join(explanations[:3])
            if len(explanations) > 3:
                explanation += f" +{len(explanations)-3} more"
        else:
            explanation = "No synonyms found"
            # Show what we tried to look up for debugging
            if debug_tokens:
                explanation += f" (tried: {', '.join(debug_tokens[:3])})"
        
        # Enhanced debug output for first few queries
        if self.debug_query_count < 3:
            print(f"Query expansion debug: processed {len(debug_tokens)} tokens")
            if len(final_terms) > 0:
                print(f"  Found {len(final_terms)} synonym terms: {final_terms[:5]}")
            else:
                print(f"  No synonyms found for any tokens")
                if debug_tokens:
                    print(f"  Tried lookups: {debug_tokens[:3]}")
            self.debug_query_count += 1
        
        return final_terms, explanation
    
    def compute_bm25_score_with_explanation(self, query_terms: List[str], doc_terms: List[str], 
                                           corpus_stats: Dict[str, Any]) -> Tuple[float, str]:
        """BM25 scoring with detailed explanation."""
        score = 0.0
        explanations = []
        doc_len = len(doc_terms)
        avgdl = corpus_stats.get('avg_doc_length', doc_len)
        
        matching_terms = []
        for term in set(query_terms):
            if term in doc_terms:
                tf = doc_terms.count(term)
                df = corpus_stats.get('doc_freq', {}).get(term, 1)
                N = corpus_stats.get('total_docs', 1)
                
                # BM25 calculation
                idf = math.log((N - df + 0.5) / (df + 0.5))
                tf_component = (tf * 1.5 + tf) / (tf + 1.5 * (1 - 0.75 + 0.75 * (doc_len / avgdl)))
                term_score = idf * tf_component
                score += term_score
                
                matching_terms.append(term)
                if len(explanations) < 3:  # Top 3 terms
                    explanations.append(f"'{term}'({term_score:.2f})")
        
        explanation = f"Matched {len(matching_terms)} terms: {'; '.join(explanations)}" if explanations else "No term matches"
        if len(matching_terms) > 3:
            explanation += f" +{len(matching_terms) - 3} more"
            
        return score, explanation
    
    def compute_semantic_similarity_with_explanation(self, req_doc: spacy.tokens.Doc, 
                                                   act_doc: spacy.tokens.Doc) -> Tuple[float, str]:
        """Semantic similarity with explanation."""
        try:
            # Try transformer embeddings first
            vec1 = req_doc._.trf_data.last_hidden_layer_state.data.mean(axis=0)
            vec2 = act_doc._.trf_data.last_hidden_layer_state.data.mean(axis=0)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            similarity = float(np.dot(vec1, vec2) / (norm1 * norm2)) if norm1 > 0 and norm2 > 0 else 0.0
            method = "transformer embeddings"
        except:
            # Fallback to standard spaCy similarity
            similarity = req_doc.similarity(act_doc)
            method = "spaCy similarity"
        
        # Determine similarity level
        if similarity >= 0.7:
            level = "Very High"
        elif similarity >= 0.5:
            level = "High"
        elif similarity >= 0.3:
            level = "Medium"
        elif similarity >= 0.1:
            level = "Low"
        else:
            level = "Very Low"
        
        explanation = f"{level} semantic similarity ({similarity:.3f}) via {method}"
        return max(0.0, similarity), explanation
    
    def compute_domain_similarity_with_explanation(self, req_terms: List[str], act_terms: List[str],
                                                 domain_weights: Dict[str, float]) -> Tuple[float, str]:
        """Domain similarity with explanation."""
        req_domain_terms = [term for term in req_terms if term in domain_weights]
        act_domain_terms = [term for term in act_terms if term in domain_weights]
        
        if not req_domain_terms or not act_domain_terms:
            return 0.0, "No domain terms found in both texts"
        
        common_domain = set(req_domain_terms) & set(act_domain_terms)
        if not common_domain:
            return 0.0, f"No shared domain terms (req: {len(req_domain_terms)}, act: {len(act_domain_terms)})"
        
        domain_score = sum(domain_weights[term] for term in common_domain)
        domain_score /= max(len(req_domain_terms), len(act_domain_terms))
        
        # Show most important shared terms
        shared_with_weights = [(term, domain_weights[term]) for term in common_domain]
        shared_with_weights.sort(key=lambda x: x[1], reverse=True)
        top_shared = shared_with_weights[:2]
        
        explanation = f"Shared {len(common_domain)} domain terms: "
        explanation += ", ".join([f"'{term}'({weight:.2f})" for term, weight in top_shared])
        if len(shared_with_weights) > 2:
            explanation += f" +{len(shared_with_weights)-2} more"
            
        return domain_score, explanation
    
    def compute_bm25_score(self, query_terms: List[str], doc_terms: List[str], 
                          corpus_stats: Dict[str, Any], k1: float = 1.5, b: float = 0.75) -> float:
        """BM25 scoring - original implementation."""
        score, _ = self.compute_bm25_score_with_explanation(query_terms, doc_terms, corpus_stats)
        return score
    
    def compute_comprehensive_similarity_with_explanation(self, req_doc: spacy.tokens.Doc, 
                                                        act_doc: spacy.tokens.Doc,
                                                        req_terms: List[str],
                                                        act_terms: List[str],
                                                        corpus_stats: Dict[str, Any],
                                                        domain_weights: Dict[str, float]) -> Tuple[Dict[str, float], Dict[str, str]]:
        """Compute all similarity scores with explanations."""
        scores = {}
        explanations = {}
        
        # 1. Dense semantic similarity
        scores['dense_semantic'], explanations['semantic'] = self.compute_semantic_similarity_with_explanation(req_doc, act_doc)
        
        # 2. BM25 similarity  
        scores['bm25'], explanations['bm25'] = self.compute_bm25_score_with_explanation(req_terms, act_terms, corpus_stats)
        
        # 3. Syntactic similarity
        req_syntax = self.extract_syntactic_features(req_doc)
        act_syntax = self.extract_syntactic_features(act_doc)
        scores['syntactic'], explanations['syntactic'] = self.compute_syntactic_similarity_with_explanation(req_syntax, act_syntax)
        
        # 4. Domain-weighted similarity
        scores['domain_weighted'], explanations['domain'] = self.compute_domain_similarity_with_explanation(req_terms, act_terms, domain_weights)
        
        # 5. Query expansion similarity
        expanded_req, query_exp = self.expand_query_with_explanation(req_doc)
        expansion_overlap = len(set(expanded_req) & set(act_terms))
        scores['query_expansion'] = expansion_overlap / max(len(expanded_req), 1)
        
        # Enhanced query expansion explanation
        if expanded_req:
            matched_expansions = list(set(expanded_req) & set(act_terms))
            explanations['query_expansion'] = f"{expansion_overlap}/{len(expanded_req)} expansion matches"
            if matched_expansions:
                explanations['query_expansion'] += f": {matched_expansions[:2]}"
            explanations['query_expansion'] += f" | {query_exp}"
        else:
            explanations['query_expansion'] = "No query expansion terms found"
        
        return scores, explanations
    
    def create_match_explanation(self, req_id: str, req_text: str, act_name: str,
                               scores: Dict[str, float], explanations: Dict[str, str],
                               weights: Dict[str, float], req_terms: List[str], act_terms: List[str]) -> MatchExplanation:
        """Create detailed match explanation."""
        combined_score = sum(weights.get(key, 0) * score for key, score in scores.items())
        
        # Determine match quality
        if combined_score >= 1.5:
            match_quality = "EXCELLENT"
        elif combined_score >= 1.0:
            match_quality = "GOOD"
        elif combined_score >= 0.7:
            match_quality = "MODERATE"
        else:
            match_quality = "WEAK"
        
        # Determine semantic similarity level
        semantic_score = scores.get('dense_semantic', 0)
        if semantic_score >= 0.7:
            semantic_level = "Very High"
        elif semantic_score >= 0.5:
            semantic_level = "High"
        elif semantic_score >= 0.3:
            semantic_level = "Medium"
        elif semantic_score >= 0.1:
            semantic_level = "Low"
        else:
            semantic_level = "Very Low"
        
        # Find shared terms
        shared_terms = list(set(req_terms) & set(act_terms))
        
        return MatchExplanation(
            requirement_id=req_id,
            requirement_text=req_text[:100] + "..." if len(req_text) > 100 else req_text,
            activity_name=act_name,
            combined_score=combined_score,
            semantic_score=scores.get('dense_semantic', 0),
            bm25_score=scores.get('bm25', 0),
            syntactic_score=scores.get('syntactic', 0),
            domain_score=scores.get('domain_weighted', 0),
            query_expansion_score=scores.get('query_expansion', 0),
            semantic_explanation=explanations.get('semantic', 'N/A'),
            bm25_explanation=explanations.get('bm25', 'N/A'),
            syntactic_explanation=explanations.get('syntactic', 'N/A'),
            domain_explanation=explanations.get('domain', 'N/A'),
            query_expansion_explanation=explanations.get('query_expansion', 'N/A'),
            shared_terms=shared_terms[:5],  # Top 5 shared terms
            semantic_similarity_level=semantic_level,
            match_quality=match_quality
        )
    
    def explain_top_matches(self, matches_df: pd.DataFrame, explanations: List[MatchExplanation], 
                           num_examples: int = 3):
        """Generate explanations for top matches."""
        print(f"\n{'='*80}")
        print(f"MATCH EXPLANATIONS - Top {num_examples} Matches")
        print(f"{'='*80}")
        
        # Get top matches by combined score
        top_explanations = sorted(explanations, key=lambda x: x.combined_score, reverse=True)[:num_examples]
        
        for idx, exp in enumerate(top_explanations, 1):
            print(f"\n--- MATCH {idx} ---")
            print(f"Requirement: {exp.requirement_text}")
            print(f"Activity: {exp.activity_name}")
            print(f"Combined Score: {exp.combined_score:.3f}")
            print(f"Match Quality: {exp.match_quality}")
            print()
            
            print(f"Semantic ({exp.semantic_score:.3f}): {exp.semantic_explanation}")
            print(f"BM25 ({exp.bm25_score:.3f}): {exp.bm25_explanation}")
            print(f"Syntactic ({exp.syntactic_score:.3f}): {exp.syntactic_explanation}")
            print(f"Domain ({exp.domain_score:.3f}): {exp.domain_explanation}")
            print(f"Query Expansion ({exp.query_expansion_score:.3f}): {exp.query_expansion_explanation}")
            
            # Shared terms
            if exp.shared_terms:
                print(f"Shared Terms: {', '.join(exp.shared_terms)}")
            else:
                print(f"Shared Terms: None")
    
    def evaluate_matches(self, matches_df: pd.DataFrame, gold_pairs: List[Tuple]) -> Dict[str, float]:
        """Evaluate matches against gold standard."""
        def normalize(text):
            text = str(text).strip()
            text = re.sub(r'^\d+(\.\d+)*\s+', '', text)
            text = text.split('(context')[0]
            text = text.strip().lower().replace("  ", " ").replace("-", " ")
            return text
        
        gold_set = set((normalize(req_id), normalize(act_name)) for req_id, act_name in gold_pairs)
        results = {}
        
        for k in [1, 3, 5, 10]:
            top_k = matches_df.groupby('ID').head(k)
            predicted_pairs = set()
            
            for _, row in top_k.iterrows():
                predicted_pairs.add((
                    normalize(row['ID']),
                    normalize(row['Activity Name'])
                ))
            
            tp = len(predicted_pairs & gold_set)
            fp = len(predicted_pairs - gold_set)
            fn = len(gold_set - predicted_pairs)
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            results[f'P@{k}'] = precision
            results[f'R@{k}'] = recall
            results[f'F1@{k}'] = f1
        
        return results
    
    def run_final_matching(self, requirements_file: str = "requirements.csv",
                          activities_file: str = "activities.csv",
                          gold_pairs_file: Optional[str] = None,
                          weights: Optional[Dict[str, float]] = None,
                          min_sim: float = 0.35,
                          top_n: int = 5,
                          out_file: str = "final_clean_matches") -> pd.DataFrame:
        """Run final matching with complete explainability."""
        # Winning semantic configuration weights
        if weights is None:
            weights = {
                'dense_semantic': 0.4,
                'bm25': 0.2,
                'syntactic': 0.2,
                'domain_weighted': 0.1,
                'query_expansion': 0.1
            }
        
        # Load gold standard if available
        gold_pairs = []
        if gold_pairs_file:
            try:
                gold_df = self._safe_read_csv(gold_pairs_file)
                for _, row in gold_df.iterrows():
                    req_id = str(row['ID']).strip()
                    satisfied_by = str(row.get('Satisfied By', '')).strip()
                    
                    for match in satisfied_by.split(","):
                        clean_match = match.split("(")[0].strip()
                        clean_match = re.sub(r'^\d+(\.\d+)*\s+', '', clean_match).strip()
                        if clean_match:
                            gold_pairs.append((req_id, clean_match))
                
                logger.info(f"Loaded {len(gold_pairs)} gold standard pairs")
            except Exception as e:
                logger.warning(f"Could not load gold standard: {e}")
        
        # Load data
        requirements_df = self._safe_read_csv(requirements_file).fillna({"Requirement Text": ""})
        activities_df = self._safe_read_csv(activities_file).fillna({"Activity Name": ""})
        
        logger.info(f"Loaded {len(requirements_df)} requirements and {len(activities_df)} activities")
        
        # Prepare corpus
        all_texts = list(requirements_df["Requirement Text"]) + list(activities_df["Activity Name"])
        all_texts = [text for text in all_texts if text.strip()]
        
        # Extract domain terms
        logger.info("Extracting domain-specific terms...")
        domain_weights = self.extract_domain_terms(all_texts)
        logger.info(f"Identified {len(domain_weights)} domain-specific terms")
        
        # Show top domain terms
        if domain_weights:
            top_domain = sorted(domain_weights.items(), key=lambda x: x[1], reverse=True)[:10]
            print(f"Top domain terms: {[f'{term}({weight:.3f})' for term, weight in top_domain]}")
        
        # Compute corpus statistics
        logger.info("Computing corpus statistics...")
        all_docs = list(self.nlp.pipe(all_texts, batch_size=32))
        all_term_lists = []
        doc_freq = Counter()
        
        for doc in all_docs:
            terms = [token.lemma_.lower() for token in doc 
                    if not token.is_stop and token.is_alpha and len(token.text) > 2]
            all_term_lists.append(terms)
            doc_freq.update(set(terms))
        
        corpus_stats = {
            'total_docs': len(all_docs),
            'avg_doc_length': np.mean([len(terms) for terms in all_term_lists]),
            'doc_freq': dict(doc_freq)
        }
        
        # Process documents
        logger.info("Processing requirements and activities...")
        req_docs = list(self.nlp.pipe(requirements_df["Requirement Text"], batch_size=32))
        act_docs = list(self.nlp.pipe(activities_df["Activity Name"], batch_size=32))
        
        # Extract term lists
        req_term_lists = []
        act_term_lists = []
        
        for doc in req_docs:
            terms = [token.lemma_.lower() for token in doc 
                    if not token.is_stop and token.is_alpha and len(token.text) > 2]
            req_term_lists.append(terms)
        
        for doc in act_docs:
            terms = [token.lemma_.lower() for token in doc 
                    if not token.is_stop and token.is_alpha and len(token.text) > 2]
            act_term_lists.append(terms)
        
        # Run matching with explanations
        logger.info("Running final clean matching with explanations...")
        matches = []
        match_explanations = []
        total_reqs = len(requirements_df)
        
        # Reset debug counters
        self.debug_bm25_count = 0
        self.debug_query_count = 0
        
        for req_idx, (req_row, req_doc, req_terms) in enumerate(zip(
            requirements_df.itertuples(), req_docs, req_term_lists)):
            
            if req_idx % 10 == 0:
                logger.info(f"Processing requirement {req_idx + 1}/{total_reqs}")
            
            req_text = getattr(req_row, 'Requirement_Text', '') or getattr(req_row, '_3', '')
            if not req_text or not str(req_text).strip():
                continue
            
            candidate_scores = []
            
            for act_idx, (act_name, act_doc, act_terms) in enumerate(zip(
                activities_df["Activity Name"], act_docs, act_term_lists)):
                
                if not act_name.strip():
                    continue
                
                # Compute all similarity scores with explanations
                sim_scores, explanations = self.compute_comprehensive_similarity_with_explanation(
                    req_doc, act_doc, req_terms, act_terms, corpus_stats, domain_weights
                )
                
                # Compute weighted combined score
                combined_score = sum(weights[score_type] * score 
                                   for score_type, score in sim_scores.items())
                
                if combined_score >= min_sim:
                    # Create explanation for this match
                    req_id = getattr(req_row, 'ID', None) or getattr(req_row, '_1', req_idx)
                    explanation = self.create_match_explanation(
                        str(req_id), req_text, act_name, sim_scores, explanations, 
                        weights, req_terms, act_terms
                    )
                    
                    candidate_scores.append({
                        'act_idx': act_idx,
                        'act_name': act_name,
                        'combined_score': combined_score,
                        'explanation': explanation,
                        **sim_scores
                    })
            
            # Sort and keep top N
            candidate_scores.sort(key=lambda x: x['combined_score'], reverse=True)
            
            for candidate in candidate_scores[:top_n]:
                matches.append({
                    "ID": getattr(req_row, 'ID', None) or getattr(req_row, '_1', req_idx),
                    "Requirement Name": getattr(req_row, 'Requirement_Name', None) or getattr(req_row, '_2', f"Req_{req_idx}"),
                    "Requirement Text": req_text,
                    "Activity Name": candidate['act_name'],
                    "Combined Score": round(candidate['combined_score'], 3),
                    "Dense Semantic": round(candidate['dense_semantic'], 3),
                    "BM25 Score": round(candidate['bm25'], 3),
                    "Syntactic Score": round(candidate['syntactic'], 3),
                    "Domain Weighted": round(candidate['domain_weighted'], 3),
                    "Query Expansion": round(candidate['query_expansion'], 3)
                })
                
                match_explanations.append(candidate['explanation'])
        
        # Save results
        matches_df = pd.DataFrame(matches)
        
        if not matches_df.empty:
            csv_file = f"results/{out_file}.csv"
            output_path = Path(csv_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            matches_df.to_csv(csv_file, index=False, encoding='utf-8')
            logger.info(f"Saved {len(matches)} matches to {csv_file}")
            
            # Save explanations to JSON
            explanations_file = f"results/{out_file}_explanations.json"
            explanations_data = []
            for exp in match_explanations:
                explanations_data.append({
                    'requirement_id': exp.requirement_id,
                    'requirement_text': exp.requirement_text,
                    'activity_name': exp.activity_name,
                    'combined_score': exp.combined_score,
                    'match_quality': exp.match_quality,
                    'scores': {
                        'semantic': exp.semantic_score,
                        'bm25': exp.bm25_score,
                        'syntactic': exp.syntactic_score,
                        'domain': exp.domain_score,
                        'query_expansion': exp.query_expansion_score
                    },
                    'explanations': {
                        'semantic': exp.semantic_explanation,
                        'bm25': exp.bm25_explanation,
                        'syntactic': exp.syntactic_explanation,
                        'domain': exp.domain_explanation,
                        'query_expansion': exp.query_expansion_explanation
                    },
                    'shared_terms': exp.shared_terms,
                    'semantic_similarity_level': exp.semantic_similarity_level
                })
            
            with open(explanations_file, 'w', encoding='utf-8') as f:
                json.dump(explanations_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved explanations to {explanations_file}")
            
            # Evaluate if gold standard available
            eval_results = None
            if gold_pairs and len(matches_df) > 0:
                eval_results = self.evaluate_matches(matches_df, gold_pairs)
                
            # Print analysis
            self._print_analysis(matches_df, weights, eval_results)
            
            # Add explanations for top matches
            if len(match_explanations) > 0:
                self.explain_top_matches(matches_df, match_explanations, num_examples=3)
        else:
            logger.warning("No matches found")
        
        return matches_df
    
    def _print_analysis(self, matches_df: pd.DataFrame, 
                       weights: Dict[str, float],
                       eval_results: Optional[Dict[str, float]] = None):
        """Print detailed analysis."""
        print(f"\n{'='*70}")
        print("FINAL CLEAN MATCHER ANALYSIS")
        print(f"{'='*70}")
        
        print(f"Total matches: {len(matches_df)}")
        print(f"Requirements matched: {len(matches_df['ID'].unique())}")
        print(f"Average matches per requirement: {len(matches_df) / len(matches_df['ID'].unique()):.1f}")
        
        print("\nWeight Configuration:")
        for component, weight in weights.items():
            print(f"  {component:15}: {weight:.3f}")
        
        print(f"\nScore Component Analysis:")
        score_cols = [
            ('Dense Semantic', 'dense_semantic'),
            ('BM25 Score', 'bm25'),
            ('Syntactic Score', 'syntactic'),
            ('Domain Weighted', 'domain_weighted'),
            ('Query Expansion', 'query_expansion'),
        ]

        for display_col, key in score_cols:
            if display_col in matches_df.columns:
                avg_score = matches_df[display_col].mean()
                weight = weights.get(key, 0)
                contribution = avg_score * weight
                print(f"  {display_col:15}: avg={avg_score:.3f}, weight={weight:.2f}, contribution={contribution:.3f}")
        
        print(f"\nCombined Score Statistics:")
        print(f"  Range: {matches_df['Combined Score'].min():.3f} - {matches_df['Combined Score'].max():.3f}")
        print(f"  Mean: {matches_df['Combined Score'].mean():.3f}")
        print(f"  Median: {matches_df['Combined Score'].median():.3f}")
        print(f"  Std Dev: {matches_df['Combined Score'].std():.3f}")
        
        if eval_results:
            print(f"\nEvaluation Against Gold Standard:")
            for metric, value in eval_results.items():
                print(f"  {metric}: {value:.3f}")
            
            # Target performance check
            f1_at_5 = eval_results.get('F1@5', 0)
            print(f"\nPerformance Assessment:")
            if f1_at_5 >= 0.21:
                print(f"  EXCELLENT: F1@5 = {f1_at_5:.3f} (target: >=0.21)")
            elif f1_at_5 >= 0.19:
                print(f"  GOOD: F1@5 = {f1_at_5:.3f} (close to target)")
            else:
                print(f"  BELOW TARGET: F1@5 = {f1_at_5:.3f} (target: >=0.21)")


def main():
    """Run final clean matcher with detailed explainability."""
    print("="*70)
    print("FINAL CLEAN MATCHER - Complete with Explainability")
    print("="*70)
    
    matcher = FinalCleanMatcher()
    
    results = matcher.run_final_matching(
        requirements_file="requirements.csv",
        activities_file="activities.csv",
        gold_pairs_file="manual_matches.csv",
        min_sim=0.35,
        top_n=5,
        out_file="final_clean_matches"
    )
    
    print(f"\nFinal Clean Matching completed!")
    print(f"Results saved to: results/final_clean_matches.csv")
    print(f"Explanations saved to: results/final_clean_matches_explanations.json")
    print(f"Total matches: {len(results)}")


if __name__ == "__main__":
    main()