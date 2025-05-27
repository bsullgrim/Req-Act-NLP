import pandas as pd
import spacy
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
import logging
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import math
import chardet  # Add this import for encoding detection

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TheoreticallyEnhancedMatcher:
    """
    Enhanced text matching system incorporating multiple theoretical approaches:
    1. Dense semantic embeddings (transformer-based)
    2. Sparse lexical features (TF-IDF)
    3. Syntactic similarity (dependency parsing)
    4. Domain-specific term weighting
    5. Query expansion via semantic neighbors
    """
    
    def __init__(self, model_name: str = "en_core_web_trf"):
        """Initialize the matcher with comprehensive NLP capabilities."""
        try:
            self.nlp = spacy.load(model_name)
            logger.info(f"Loaded spaCy model: {model_name}")
        except OSError:
            logger.error(f"Could not load model {model_name}")
            raise
        
        # Initialize TF-IDF vectorizer for sparse features
        self.tfidf = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            stop_words='english',
            lowercase=True,
            token_pattern=r'\b[a-zA-Z][a-zA-Z]+\b'
        )
        
        # Cache for computed features
        self._doc_cache = {}
        self._tfidf_fitted = False
    
    def _detect_encoding(self, file_path: str) -> str:
        """Detect the encoding of a file."""
        with open(file_path, 'rb') as f:
            raw_data = f.read()
            result = chardet.detect(raw_data)
            encoding = result['encoding']
            confidence = result['confidence']
            logger.info(f"Detected encoding for {file_path}: {encoding} (confidence: {confidence:.2f})")
            return encoding

    def _safe_read_csv(self, file_path: str, **kwargs) -> pd.DataFrame:
        """Safely read CSV with automatic encoding detection."""
        encodings_to_try = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
        
        # First try to detect encoding
        try:
            detected_encoding = self._detect_encoding(file_path)
            if detected_encoding:
                encodings_to_try.insert(0, detected_encoding)
        except Exception as e:
            logger.warning(f"Could not detect encoding for {file_path}: {e}")
        
        # Try each encoding
        for encoding in encodings_to_try:
            try:
                logger.info(f"Trying to read {file_path} with encoding: {encoding}")
                df = pd.read_csv(file_path, encoding=encoding, **kwargs)
                logger.info(f"Successfully read {file_path} with encoding: {encoding}")
                return df
            except UnicodeDecodeError as e:
                logger.warning(f"Failed to read with {encoding}: {e}")
                continue
            except Exception as e:
                logger.error(f"Unexpected error reading {file_path} with {encoding}: {e}")
                continue
        
        # If all encodings fail, try with error handling
        try:
            logger.info(f"Trying to read {file_path} with UTF-8 and error handling")
            df = pd.read_csv(file_path, encoding='utf-8', encoding_errors='replace', **kwargs)
            logger.warning(f"Read {file_path} with UTF-8 but some characters may be corrupted")
            return df
        except Exception as e:
            raise RuntimeError(f"Could not read {file_path} with any encoding method: {e}")
    
    def extract_syntactic_features(self, doc: spacy.tokens.Doc) -> Dict[str, Any]:
        """
        Extract syntactic features from document for structural similarity.
        
        Theory: Syntactic structure can indicate functional similarity even
        when lexical content differs (e.g., "implement solution" vs "deploy system").
        """
        features = {
            'dep_patterns': [],  # dependency relation patterns
            'pos_sequence': [],  # POS tag sequences
            'entity_types': [],  # named entity types
            'verb_frames': [],   # verb subcategorization frames
        }
        
        # Extract dependency patterns (head -> dependent relationships)
        for token in doc:
            if not token.is_stop and token.is_alpha:
                pattern = f"{token.dep_}:{token.pos_}"
                features['dep_patterns'].append(pattern)
                
                # Capture verb frames (verb + its argument structure)
                if token.pos_ == "VERB":
                    children = [child.dep_ for child in token.children]
                    if children:
                        features['verb_frames'].append(f"{token.lemma_}:{':'.join(sorted(children))}")
        
        # POS n-grams for structural similarity
        pos_tags = [token.pos_ for token in doc if not token.is_space]
        for i in range(len(pos_tags) - 1):
            features['pos_sequence'].append(f"{pos_tags[i]}_{pos_tags[i+1]}")
        
        # Entity types (semantic categories)
        features['entity_types'] = [ent.label_ for ent in doc.ents]
        
        return features
    
    def compute_syntactic_similarity(self, features1: Dict, features2: Dict) -> float:
        """
        Compute syntactic similarity based on structural features.
        
        Theory: Jaccard similarity over syntactic features captures
        structural parallelism independent of lexical choice.
        """
        total_sim = 0.0
        weights = {
            'dep_patterns': 0.4,
            'pos_sequence': 0.3, 
            'entity_types': 0.2,
            'verb_frames': 0.1
        }
        
        for feature_type, weight in weights.items():
            set1 = set(features1.get(feature_type, []))
            set2 = set(features2.get(feature_type, []))
            
            if not set1 and not set2:
                jaccard = 1.0  # Both empty
            elif not set1 or not set2:
                jaccard = 0.0  # One empty
            else:
                intersection = len(set1 & set2)
                union = len(set1 | set2)
                jaccard = intersection / union if union > 0 else 0.0
            
            total_sim += weight * jaccard
        
        return total_sim
    
    def extract_domain_terms(self, corpus: List[str]) -> Dict[str, float]:
        """
        Extract domain-specific terms and compute their importance weights.
        
        Theory: Domain-specific terms should be weighted higher than
        general vocabulary when computing relevance scores.
        """
        # Combine all text for domain analysis
        all_text = ' '.join(corpus)
        doc = self.nlp(all_text)
        
        # Extract candidate terms (noun phrases, technical terms)
        candidates = []
        for chunk in doc.noun_chunks:
            if len(chunk.text.split()) <= 3:  # Limit to reasonable phrase length
                candidates.append(chunk.text.lower().strip())
        
        # Add individual technical terms (capitalized words, compound terms)
        for token in doc:
            if (token.pos_ in ['NOUN', 'PROPN'] and 
                not token.is_stop and 
                len(token.text) > 3):
                candidates.append(token.lemma_.lower())
        
        # Compute term frequencies and filter for domain relevance
        term_freq = Counter(candidates)
        total_terms = sum(term_freq.values())
        
        # Weight terms by frequency and length (longer terms often more specific)
        domain_weights = {}
        for term, freq in term_freq.items():
            if freq >= 2:  # Must appear at least twice
                # Combine frequency and length weighting
                length_bonus = min(len(term.split()), 3) / 3.0
                freq_weight = freq / total_terms
                domain_weights[term] = freq_weight * (1 + length_bonus)
        
        # Normalize weights
        max_weight = max(domain_weights.values()) if domain_weights else 1.0
        return {term: weight / max_weight for term, weight in domain_weights.items()}
    
    def expand_query(self, query_doc: spacy.tokens.Doc, k: int = 5) -> List[str]:
        """
        Expand query with semantically similar terms.
        
        Theory: Query expansion addresses vocabulary mismatch problem
        by including semantically related terms that might appear in relevant documents.
        """
        expanded_terms = []
        
        # Get most similar vectors for key terms
        for token in query_doc:
            if (token.pos_ in ['NOUN', 'VERB', 'ADJ'] and 
                not token.is_stop and 
                token.has_vector and
                token.vector_norm > 0):
                
                # Find similar words in vocabulary
                try:
                    ms = self.nlp.vocab.vectors.most_similar(
                        np.asarray([token.vector]), n=k
                    )
                    
                    for score, similar_key in zip(ms[0], ms[1]):
                        if score > 0.7:  # High similarity threshold
                            similar_word = self.nlp.vocab.strings[similar_key]
                            if similar_word != token.text.lower():
                                expanded_terms.append(similar_word)
                except Exception as e:
                    logger.debug(f"Could not find similar words for {token.text}: {e}")
                    continue
        
        return list(set(expanded_terms))
    
    def compute_bm25_score(self, query_terms: List[str], doc_terms: List[str], 
                          corpus_stats: Dict[str, Any], k1: float = 1.5, b: float = 0.75) -> float:
        """
        Implement BM25 scoring for better term frequency handling.
        
        Theory: BM25 addresses issues with TF-IDF by using saturation functions
        and length normalization, performing better for shorter queries.
        """
        score = 0.0
        doc_len = len(doc_terms)
        avgdl = corpus_stats.get('avg_doc_length', doc_len)
        
        for term in set(query_terms):
            if term in doc_terms:
                tf = doc_terms.count(term)
                df = corpus_stats.get('doc_freq', {}).get(term, 1)
                N = corpus_stats.get('total_docs', 1)
                
                # IDF component
                idf = math.log((N - df + 0.5) / (df + 0.5))
                
                # TF component with saturation
                tf_component = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (doc_len / avgdl)))
                
                score += idf * tf_component
        
        return score
    
    def compute_comprehensive_similarity(self, req_doc: spacy.tokens.Doc, 
                                       act_doc: spacy.tokens.Doc,
                                       req_terms: List[str],
                                       act_terms: List[str],
                                       corpus_stats: Dict[str, Any],
                                       domain_weights: Dict[str, float]) -> Dict[str, float]:
        """
        Compute multiple similarity scores using different theoretical approaches.
        """
        scores = {}
        
        # 1. Dense semantic similarity (transformer embeddings)
        try:
            vec1 = req_doc._.trf_data.last_hidden_layer_state.data.mean(axis=0)
            vec2 = act_doc._.trf_data.last_hidden_layer_state.data.mean(axis=0)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            scores['dense_semantic'] = float(np.dot(vec1, vec2) / (norm1 * norm2)) if norm1 > 0 and norm2 > 0 else 0.0
        except:
            scores['dense_semantic'] = req_doc.similarity(act_doc)
        
        # 2. Sparse lexical similarity (BM25)
        scores['bm25'] = self.compute_bm25_score(req_terms, act_terms, corpus_stats)
        
        # 3. Syntactic similarity
        req_syntax = self.extract_syntactic_features(req_doc)
        act_syntax = self.extract_syntactic_features(act_doc)
        scores['syntactic'] = self.compute_syntactic_similarity(req_syntax, act_syntax)
        
        # 4. Domain-weighted similarity
        domain_score = 0.0
        req_domain_terms = [term for term in req_terms if term in domain_weights]
        act_domain_terms = [term for term in act_terms if term in domain_weights]
        
        if req_domain_terms and act_domain_terms:
            common_domain = set(req_domain_terms) & set(act_domain_terms)
            if common_domain:
                domain_score = sum(domain_weights[term] for term in common_domain)
                domain_score /= max(len(req_domain_terms), len(act_domain_terms))
        
        scores['domain_weighted'] = domain_score
        
        # 5. Query expansion similarity
        expanded_req = self.expand_query(req_doc)
        expansion_overlap = len(set(expanded_req) & set(act_terms))
        scores['query_expansion'] = expansion_overlap / max(len(expanded_req), 1)
        
        return scores
    
    def run_enhanced_matcher(self, 
                           requirements_file: str = "requirements.csv",
                           activities_file: str = "activities.csv",
                           weights: Optional[Dict[str, float]] = None,
                           min_sim: float = 0.3,
                           top_n: int = 5,
                           out_file: str = "enhanced_matches") -> pd.DataFrame:
        """
        Run the theoretically enhanced matching algorithm.
        """
        # Default weights based on IR theory
        if weights is None:
            weights = {
                'dense_semantic': 0.3,    # Good for topical similarity
                'bm25': 0.25,            # Strong for exact term matches
                'syntactic': 0.2,        # Captures structural similarity
                'domain_weighted': 0.15, # Emphasizes technical terms
                'query_expansion': 0.1   # Handles vocabulary mismatch
            }
        
        # Validate weights sum to 1
        weight_sum = sum(weights.values())
        if abs(weight_sum - 1.0) > 0.001:
            logger.warning(f"Weights sum to {weight_sum}, normalizing...")
            weights = {k: v/weight_sum for k, v in weights.items()}
        
        # Load data with safe encoding detection
        try:
            requirements_df = self._safe_read_csv(requirements_file).fillna({"Requirement Text": ""})
            activities_df = self._safe_read_csv(activities_file).fillna({"Activity Name": ""})
        except Exception as e:
            logger.error(f"Failed to load CSV files: {e}")
            raise
        
        logger.info(f"Loaded {len(requirements_df)} requirements and {len(activities_df)} activities")
        
        # Prepare corpus for analysis
        all_texts = list(requirements_df["Requirement Text"]) + list(activities_df["Activity Name"])
        all_texts = [text for text in all_texts if text.strip()]
        
        # Extract domain-specific terms
        logger.info("Extracting domain-specific terms...")
        domain_weights = self.extract_domain_terms(all_texts)
        logger.info(f"Identified {len(domain_weights)} domain-specific terms")
        
        # Compute corpus statistics for BM25
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
        
        # Run matching
        matches = []
        total_reqs = len(requirements_df)
        
        for req_idx, (req_row, req_doc, req_terms) in enumerate(zip(
            requirements_df.itertuples(), req_docs, req_term_lists)):
            
            if req_idx % 10 == 0:
                logger.info(f"Processing requirement {req_idx + 1}/{total_reqs}")
            
            req_text = getattr(req_row, 'Requirement_Text', '') or getattr(req_row, '_2', '')
            if not req_text or not str(req_text).strip():
                continue
            
            candidate_scores = []
            
            for act_idx, (act_name, act_doc, act_terms) in enumerate(zip(
                activities_df["Activity Name"], act_docs, act_term_lists)):
                
                if not act_name.strip():
                    continue
                
                # Compute all similarity scores
                sim_scores = self.compute_comprehensive_similarity(
                    req_doc, act_doc, req_terms, act_terms, corpus_stats, domain_weights
                )
                
                # Compute weighted combined score
                combined_score = sum(weights[score_type] * score 
                                   for score_type, score in sim_scores.items())
                
                if combined_score >= min_sim:
                    candidate_scores.append({
                        'act_idx': act_idx,
                        'act_name': act_name,
                        'combined_score': combined_score,
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
        
        # Save results with safe encoding
        matches_df = pd.DataFrame(matches)
        
        if not matches_df.empty:
            csv_file = f"{out_file}.csv"
            
            # Create directory if it doesn't exist
            output_path = Path(csv_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            try:
                matches_df.to_csv(csv_file, index=False, encoding='utf-8')
                logger.info(f"Saved {len(matches)} matches to {csv_file}")
            except Exception as e:
                logger.warning(f"Failed to save with UTF-8, trying latin-1: {e}")
                try:
                    matches_df.to_csv(csv_file, index=False, encoding='latin-1')
                    logger.info(f"Saved {len(matches)} matches to {csv_file} with latin-1 encoding")
                except Exception as e2:
                    logger.error(f"Failed to save CSV file: {e2}")
                    raise
            
            # Print detailed analysis
            self._print_detailed_analysis(matches_df, weights)
        else:
            logger.warning("No matches found with current parameters")
        
        return matches_df
    
    def _print_detailed_analysis(self, matches_df: pd.DataFrame, weights: Dict[str, float]):
        """Print detailed analysis of matching results."""
        print(f"\n{'='*70}")
        print("THEORETICALLY ENHANCED MATCHING ANALYSIS")
        print(f"{'='*70}")
        
        print(f"Total matches: {len(matches_df)}")
        print(f"Requirements matched: {len(matches_df['ID'].unique())}")
        print(f"Average matches per requirement: {len(matches_df) / len(matches_df['ID'].unique()):.1f}")
        
        print("\nScore Component Analysis:")
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


def main():
    """Example usage with different theoretical configurations."""
    matcher = TheoreticallyEnhancedMatcher()
    
    # Configuration emphasizing semantic understanding
    semantic_config = {
        'weights': {
            'dense_semantic': 0.4,
            'bm25': 0.2,
            'syntactic': 0.2,
            'domain_weighted': 0.1,
            'query_expansion': 0.1
        },
        'min_sim': 0.35,
        'top_n': 5,
        'out_file': 'results/semantic_focused'
    }
    
    # Configuration emphasizing lexical precision
    lexical_config = {
        'weights': {
            'dense_semantic': 0.2,
            'bm25': 0.4,
            'syntactic': 0.15,
            'domain_weighted': 0.2,
            'query_expansion': 0.05
        },
        'min_sim': 0.3,
        'top_n': 3,
        'out_file': 'results/lexical_focused'
    }
    
    # Balanced configuration
    balanced_config = {
        'weights': {
            'dense_semantic': 0.3,
            'bm25': 0.25,
            'syntactic': 0.2,
            'domain_weighted': 0.15,
            'query_expansion': 0.1
        },
        'min_sim': 0.3,
        'top_n': 5,
        'out_file': 'results/balanced'
    }
    
    configs = [semantic_config, lexical_config, balanced_config]
    
    for i, config in enumerate(configs, 1):
        print(f"\n{'='*60}")
        print(f"RUNNING CONFIGURATION {i}: {config['out_file'].split('/')[-1].upper()}")
        print(f"{'='*60}")
        
        try:
            results = matcher.run_enhanced_matcher(**config)
            print(f"Configuration {i} completed successfully")
        except Exception as e:
            logger.error(f"Configuration {i} failed: {e}")


if __name__ == "__main__":
    main()