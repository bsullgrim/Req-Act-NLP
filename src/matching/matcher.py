"""
Aerospace Requirements Matcher - Clean Version
Focused matching system for aerospace domain with proper separation of concerns.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Any, Optional, Set
import logging
from pathlib import Path
from collections import Counter
import math
import spacy
import json
import re
import torch
from dataclasses import dataclass

# Import existing utils
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# Import existing utils
from src.utils.file_utils import SafeFileHandler
from src.utils.path_resolver import SmartPathResolver
from src.utils.repository_setup import RepositoryStructureManager
from src.matching.domain_resources import DomainResources
# Optional dependencies
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = False
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

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

class AerospaceMatcher:
    """
    Aerospace-optimized requirements matcher with domain-specific enhancements.
    """    
    def __init__(self, model_name: str = "en_core_web_lg", repo_manager=None):
        """Initialize aerospace matcher with enhanced NLP capabilities."""
        
        # Load spaCy model
        try:
            self.nlp = spacy.load(model_name)
            logger.info(f"✅ Loaded spaCy model: {model_name}")
        except OSError:
            logger.error(f"❌ Could not load model {model_name}")
            raise
        
        # Repository management
        if repo_manager is None:
            raise ValueError("Repository manager is required")
        self.repo_manager = repo_manager
        
        # Initialize utilities
        self.file_handler = SafeFileHandler(repo_manager)
        self.path_resolver = SmartPathResolver(repo_manager)
        self.domain = DomainResources()

        # Initialize enhanced semantic similarity if available
        self._setup_semantic_model()
        
        # Load aerospace knowledge
        self.all_aerospace_terms = self.domain.get_domain_terms()
        self.synonyms = self.domain.synonyms

        
        # Performance caches
        self.preprocessing_cache = {}
        self.semantic_cache = {}
        
        logger.info(f"🚀 Aerospace matcher initialized with {len(self.all_aerospace_terms)} domain terms")
    
    def _setup_semantic_model(self):
        """Setup semantic similarity model with speed optimization."""
        
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                # Check GPU availability first
                if torch.cuda.is_available():
                    device = "cuda"
                    gpu_name = torch.cuda.get_device_name(0)
                    print(f"🚀 GPU detected: {gpu_name}")
                    
                    # GPU models (prioritize speed with good quality)
                    model_options = [
                        'all-MiniLM-L6-v2',      # Fast and good quality (22MB)
                        'all-MiniLM-L12-v2',     # Better quality, still fast (33MB)
                        'all-mpnet-base-v2'      # Best quality, slower (420MB)
                    ]
                else:
                    device = "cpu"
                    print("💻 Using CPU (selecting fastest model)")
                    
                    # CPU - use smallest/fastest model
                    model_options = [
                        'all-MiniLM-L6-v2',      # Fastest option
                    ]
            
                for model_name in model_options:
                    try:
                        self.semantic_model = SentenceTransformer(model_name, device=device)
                        # Test the model
                        test_embedding = self.semantic_model.encode(["test"], show_progress_bar=False)
                        print(f"✅ Loaded semantic model: {model_name} on {device}")
                        logger.info(f"✅ Loaded semantic model: {model_name}")
                        self.use_enhanced_semantic = True
                        return
                    except Exception as e:
                        logger.warning(f"⚠️ Could not load {model_name}: {e}")
                        continue
            
                # If all models fail
                self.use_enhanced_semantic = False
                logger.warning("⚠️ No semantic models available, using spaCy vectors only")
            
            except Exception as e:
                logger.warning(f"⚠️ Sentence transformers setup failed: {e}")
                self.use_enhanced_semantic = False
        else:
            self.use_enhanced_semantic = False
            logger.info("📦 Install sentence-transformers for enhanced semantic matching")

    def load_requirements(self, path: str) -> pd.DataFrame:
        """Load requirements data with proper error handling (FIXED empty requirements issue)."""
        df = self.file_handler.safe_read_csv(path)
        
        # Handle empty requirements (addressing user's issue)
        original_count = len(df)
        df = df.dropna(subset=['Requirement Text'])
        df = df[df['Requirement Text'].str.strip() != '']
        
        if len(df) < original_count:
            removed_count = original_count - len(df)
            logger.warning(f"⚠️ Removed {removed_count} empty requirements")
        
        return df.fillna({"Requirement Text": "", "ID": "", "Requirement Name": ""})
    
    def load_activities(self, path: str) -> pd.DataFrame:
        """Load activities data with duplicate removal (FIXED main duplicate issue)."""
        df = self.file_handler.safe_read_csv(path)
        
        # Handle empty activities
        original_count = len(df)
        df = df.dropna(subset=['Activity Name'])
        df = df[df['Activity Name'].str.strip() != '']
        
        # Remove duplicates (addressing user's main F1 score issue)
        df_deduped = df.drop_duplicates(subset=['Activity Name'], keep='first')
        
        if len(df_deduped) < len(df):
            duplicate_count = len(df) - len(df_deduped)
            logger.warning(f"⚠️ Removed {duplicate_count} duplicate activities")
        
        if len(df_deduped) < original_count:
            removed_count = original_count - len(df_deduped)
            logger.info(f"📊 Cleaned activities: {original_count} → {len(df_deduped)} ({removed_count} removed)")
        
        return df_deduped.fillna({"Activity Name": ""})
          
    def _expand_aerospace_abbreviations(self, text: str) -> str:
        """Expand common aerospace abbreviations in text using domain resources."""
        text_lower = text.lower()
        
        # Get abbreviations from domain resources
        for abbr, full_form in self.domain.abbreviations.items():
            pattern = r'\b' + re.escape(abbr) + r'\b'
            text_lower = re.sub(pattern, full_form, text_lower)
        
        return text_lower
    
    def _preprocess_text_aerospace(self, text: str) -> List[str]:
        """Aerospace-optimized text preprocessing with caching."""
        if text in self.preprocessing_cache:
            return self.preprocessing_cache[text]
        
        # Expand abbreviations
        expanded_text = self._expand_aerospace_abbreviations(text)
        
        # Process with spaCy
        doc = self.nlp(expanded_text)
        terms = []
        
        for token in doc:
            # Standard filtering with aerospace adjustments
            if (not token.is_stop and 
                not token.is_punct and 
                len(token.text) > 1 and  # Allow 2-letter aerospace terms
                token.is_alpha):
                
                # Prioritize aerospace terms
                if token.lemma_.lower() in self.all_aerospace_terms:
                    terms.append(token.lemma_.lower())
                elif token.pos_ in ['NOUN', 'VERB', 'ADJ', 'PROPN']:
                    terms.append(token.lemma_.lower())
            
            # Technical patterns (numbers, acronyms)
            if re.match(r'\d+\w+', token.text):  # "5ms", "100MHz"
                terms.append(token.text.lower())
            
            if token.text.isupper() and len(token.text) >= 2:  # Acronyms
                terms.append(token.text.lower())
        
        # Add aerospace-specific bigrams
        if len(terms) > 1:
            for i in range(len(terms) - 1):
                if (terms[i] in self.all_aerospace_terms or 
                    terms[i+1] in self.all_aerospace_terms):
                    bigram = f"{terms[i]}_{terms[i+1]}"
                    terms.append(bigram)
        
        # Cache result
        self.preprocessing_cache[text] = terms
        return terms

    def extract_domain_weights(self, corpus: List[str]) -> Dict[str, float]:
        """Extract aerospace domain term weights using TF-IDF or fallback method."""
        
        if not SKLEARN_AVAILABLE:
            logger.warning("📦 sklearn not available, using fallback domain extraction")
            return self._fallback_domain_weights(corpus)
        
        try:
            # Preprocess corpus for aerospace
            processed_corpus = [self._expand_aerospace_abbreviations(text) for text in corpus]
            
            # TF-IDF configuration for technical domains
            vectorizer = TfidfVectorizer(  
                max_features=1000,
                ngram_range=(1, 3),
                min_df=2,
                max_df=0.8,
                token_pattern=r'\b[a-zA-Z0-9][a-zA-Z0-9_\-\.\/]*[a-zA-Z0-9]\b',
                lowercase=True,
                stop_words='english'
            )
            
            tfidf_matrix = vectorizer.fit_transform(processed_corpus)
            feature_names = vectorizer.get_feature_names_out()
            avg_tfidf = tfidf_matrix.mean(axis=0).A1
            
            # Create domain weights with aerospace boosting
            domain_weights = {}
            for term, weight in zip(feature_names, avg_tfidf):
                if len(term) >= 2 and weight > 0.01:
                    boost = 1.0
                    
                    # Boost aerospace terms
                    if any(aero_term in term.lower() for aero_term in self.all_aerospace_terms):
                        boost *= 2.0
                    
                    # Boost technical patterns
                    if any(char.isdigit() for char in term):
                        boost *= 1.3
                    
                    if '_' in term or len(term.split()) > 1:
                        boost *= 1.2
                    
                    domain_weights[term] = weight * boost
            
            # Normalize weights
            if domain_weights:
                max_weight = max(domain_weights.values())
                domain_weights = {term: weight / max_weight 
                                 for term, weight in domain_weights.items()}
            
            logger.info(f"📊 Extracted {len(domain_weights)} domain terms")
            return domain_weights
            
        except Exception as e:
            logger.error(f"❌ TF-IDF extraction failed: {e}")
            return self._fallback_domain_weights(corpus)
    
    def _fallback_domain_weights(self, corpus: List[str]) -> Dict[str, float]:
        """Fallback domain extraction using predefined aerospace weights."""
        domain_weights = {}
        
        # Category-based weights
        category_weights = {
            'requirements': 0.9,
            'operations': 0.8,
            'systems': 0.7,
            'components': 0.6,
            'data': 0.5,
            'ground': 0.5,
            'flight': 0.6,
            'power': 0.5,
            'thermal': 0.5
        }
        
        for category, terms in self.AEROSPACE_TERMS.items():
            weight = category_weights.get(category, 0.5)
            for term in terms:
                domain_weights[term] = weight
        
        logger.info(f"📊 Using fallback domain weights: {len(domain_weights)} terms")
        return domain_weights
    
    def compute_semantic_similarity(self, req_doc, act_doc, req_idx=None, act_idx=None):
        """
        ENHANCED: Compute semantic similarity with intelligent fallbacks.
        """
        
        # Try enhanced semantic model first
        if self.use_enhanced_semantic:
            req_text = req_doc.text
            act_text = act_doc.text
            
            cache_key = (req_text, act_text)
            if cache_key in self.semantic_cache:
                return self.semantic_cache[cache_key]
            
            try:
                embeddings = self.semantic_model.encode([req_text, act_text], batch_size=2, show_progress_bar=False)
                similarity = float(np.dot(embeddings[0], embeddings[1]) / 
                                (np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])))
                similarity = max(0, (similarity + 1) / 2)
                
                explanation = f"Enhanced semantic: {similarity:.3f}"
                result = (similarity, explanation)
                self.semantic_cache[cache_key] = result
                return result
                
            except Exception as e:
                logger.warning(f"Semantic model error: {e}")
        
        # ENHANCED FALLBACK: Check if spaCy model has vectors
        if self.nlp.meta.get('vectors', {}).get('width', 0) > 0:
            # spaCy has vectors, use them
            try:
                similarity = req_doc.similarity(act_doc)
                explanation = f"spaCy semantic: {similarity:.3f}"
                return similarity, explanation
            except Exception as e:
                logger.warning(f"spaCy similarity error: {e}")
        
        # FINAL FALLBACK: Text-based similarity when no vectors available
        similarity = self._compute_text_similarity(req_doc.text, act_doc.text)
        explanation = f"Text-based similarity: {similarity:.3f}"
        return similarity, explanation    
    
    def compute_bm25_score(self, req_terms: List[str], act_terms: List[str],
                        corpus_stats: Dict[str, Any]) -> Tuple[float, str]:
        """Compute BM25 score with aerospace term boosting."""
        
        score = 0.0
        doc_len = len(act_terms)
        avgdl = corpus_stats.get('avg_doc_length', doc_len)
        N = corpus_stats.get('total_docs', 1)
        
        # BM25 parameters - TUNED FOR BETTER PERFORMANCE
        k1 = 1.5  # Increased from 1.2 for better term frequency saturation
        b = 0.3   # Increased from 0.1 for less document length penalty
        
        matching_terms = []
        aerospace_matches = []
        
        for term in set(req_terms):
            if term in act_terms:
                tf = act_terms.count(term)
                df = corpus_stats.get('doc_freq', {}).get(term, 1)
                
                # BM25 formula
                idf = math.log((N - df + 0.5) / (df + 0.5))
                tf_component = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (doc_len / avgdl)))
                term_score = idf * tf_component
                
                # Boost aerospace terms (but less aggressively)
                if term in self.all_aerospace_terms:
                    term_score *= 1.3  # Reduced from 1.5
                    aerospace_matches.append(term)
                
                score += term_score
                matching_terms.append(term)
        
        # BOOST for short activities with exact matches (moved OUTSIDE the loop!)
        if len(act_terms) <= 3:
            exact_matches = len(set(req_terms) & set(act_terms))
            if exact_matches > 0:
                score *= (1 + 0.3 * exact_matches)  # Reduced from 0.5 to 0.3
        
        # Normalize by theoretical maximum with coverage consideration
        if req_terms:
            # Add coverage component
            coverage = len(matching_terms) / len(set(req_terms))
            normalized_score = min(1.0, (score / 10.0) * (0.8 + 0.2 * coverage))
        else:
            normalized_score = 0
        
        # Create explanation
        explanation = f"BM25: {len(matching_terms)} matches"
        if aerospace_matches:
            explanation += f" [aerospace: {', '.join(aerospace_matches[:2])}]"
        
        return normalized_score, explanation
    
    def compute_domain_similarity(self, req_terms: List[str], act_terms: List[str], 
                                domain_weights: Dict[str, float]) -> Tuple[float, Dict]:
        """
        Compute domain-specific similarity with rich structured explanation.
        Returns:
            final_score: float
            explanation: dict with detailed evidence
        """
        score = 0.0
        explanation = {
            "aerospace_terms": [],
            "key_indicators": {},
            "learned_relationships": {},
            "phrases": [],
            "weighted_terms": {},
            "multi_evidence_bonus": 0,
            "resource_provenance": {},
            "final_score": 0.0
        }

        req_set = set(req_terms)
        act_set = set(act_terms)

        # 1. Aerospace vocabulary overlap
        req_aero = req_set & self.all_aerospace_terms
        act_aero = act_set & self.all_aerospace_terms
        shared_aero = req_aero & act_aero
        if shared_aero:
            vocab_score = 0.3 + (0.1 * min(len(shared_aero)-1, 3))
            score += vocab_score
            explanation["aerospace_terms"] = list(shared_aero)
        elif req_aero or act_aero:
            score += 0.1

        # 2. Key indicators
        if hasattr(self.domain, "get_strong_indicator_words"):
            strong_indicators = self.domain.get_strong_indicator_words()
            indicator_matches = req_set & act_set & strong_indicators
            if indicator_matches:
                indicator_score = 0.0
                for term in indicator_matches:
                    weight = 0.05
                    rules = getattr(self.domain, "matching_rules", {}).get("strong_indicators", {})
                    rule_key = f"shared_word:{term}"
                    if rule_key in rules:
                        weight = min(0.1 * (rules[rule_key] / 10), 0.15)
                    indicator_score += weight
                    explanation["key_indicators"][term] = {"weight": weight, "source": "matching_rules"}
                score += indicator_score

        # 3. Learned term relationships
        if hasattr(self.domain, "get_domain_cooccurrence_score"):
            co_score = self.domain.get_domain_cooccurrence_score(req_set, act_set)
            if co_score > 0:
                score += co_score * 0.4
                # capture which terms contributed
                relationships = {}
                for req_term in req_set:
                    co_terms = set(self.domain.cooccurrence_terms.get(req_term, [])) & act_set
                    if co_terms:
                        relationships[req_term] = list(co_terms)
                explanation["learned_relationships"] = relationships

        # 4. Phrase patterns
        matched_phrases = []
        if hasattr(self.domain, "phrase_patterns"):
            phrase_score = 0.0
            for ph_type in ["requirement_patterns", "activity_patterns"]:
                patterns = self.domain.phrase_patterns.get(ph_type, {})
                for phrase, count in patterns.items():
                    if count >= 2 and len(matched_phrases) < 3:
                        words = set(phrase.split())
                        if words.issubset(req_set) and words.issubset(act_set):
                            phrase_score += 0.1
                            matched_phrases.append({"phrase": phrase, "type": ph_type, "frequency": count})
            if phrase_score > 0:
                score += min(phrase_score, 0.4)
                explanation["phrases"] = matched_phrases

        # 5. Weighted domain terms
        req_domain_terms = [t for t in req_terms if t in domain_weights]
        act_domain_terms = [t for t in act_terms if t in domain_weights]
        common_domain = set(req_domain_terms) & set(act_domain_terms)
        weighted_terms = {}
        if common_domain:
            domain_score = 0.0
            for term in common_domain:
                w = domain_weights[term]
                boosts = []
                if hasattr(self.domain, "get_strong_indicator_words") and term in strong_indicators:
                    w *= 1.5
                    boosts.append("strong_indicator")
                if term in self.all_aerospace_terms:
                    w *= 1.2
                    boosts.append("aerospace_term")
                domain_score += w
                weighted_terms[term] = {"weight": w, "boosts": boosts}
            norm_factor = max(len(req_domain_terms), len(act_domain_terms))
            domain_score = domain_score / norm_factor if norm_factor > 0 else 0
            score += min(domain_score, 0.3)
            explanation["weighted_terms"] = weighted_terms

        # 6. Multi-evidence bonus
        evidence_types = sum(1 for k in ["aerospace_terms","key_indicators","learned_relationships","phrases","weighted_terms"] 
                            if explanation[k])
        if evidence_types >= 3:
            bonus = 0.1 + (0.05 * (evidence_types - 3))
            score += bonus
            explanation["multi_evidence_bonus"] = bonus

        # 7. Resource provenance
        explanation["resource_provenance"] = self.domain.get_resource_status()

        # 8. Final score
        final_score = min(score, 1.0)
        explanation["final_score"] = final_score

        return final_score, explanation
   
    def expand_query_aerospace(self, query_terms: List[str], activity_terms: List[str]) -> Tuple[float, str]:
        """
        UNIFIED: Query expansion using aerospace synonyms with activity expansion.
        
        Strategy:
        1. Expand activity terms with synonyms (helps sparse activities match requirements)
        2. Calculate overlap between requirement terms and expanded activities
        3. Score based on how well the expanded activity covers requirement terms
        
        Args:
            query_terms: Terms extracted from requirement
            activity_terms: Terms extracted from activity
            
        Returns:
            Tuple of (score, explanation)
        """
        
        if not query_terms or not activity_terms:
            return 0.0, "No terms to expand"
        
        # Normalize requirement terms
        req_terms_set = set(term.lower().strip() for term in query_terms if term.strip())
        
        # Start with original activity terms
        expanded_activity_terms = set(term.lower().strip() for term in activity_terms if term.strip())
        
        # Expand activity terms using domain resources
        expansion_count = 0
        for term in activity_terms:
            term_lower = term.lower().strip()
            if term_lower:
                # Get synonyms from domain resources
                synonyms = self.domain.get_synonyms(term_lower)
                for synonym in synonyms:
                    if synonym.strip():
                        expanded_activity_terms.add(synonym.lower().strip())
                        expansion_count += 1
        
        # Calculate overlap: requirement terms found in expanded activities
        overlap_terms = req_terms_set & expanded_activity_terms
        overlap_count = len(overlap_terms)
        
        # Score: What fraction of requirement terms are addressed by expanded activity?
        score = overlap_count / len(req_terms_set) if req_terms_set else 0.0
        
        # Build details dictionary
        details = {
            "explanation": f"Activity expansion: {overlap_count}/{len(req_terms_set)} req terms matched"
                        + (f" (expanded {expansion_count} synonyms)" if expansion_count else ""),
            "requirement_terms": list(req_terms_set),
            "expanded_activity_terms": list(expanded_activity_terms),
            "matched_terms": list(overlap_terms)
        }

        return score, details
    
    def compute_comprehensive_similarity(self, req_doc, act_doc, req_terms: List[str],
                                    act_terms: List[str], corpus_stats: Dict[str, Any],
                                    domain_weights: Dict[str, float],
                                    req_idx: int = None, act_idx: int = None) -> Tuple[Dict[str, float], Dict[str, str]]:
        """ENHANCED: Compute all similarity components with optional precomputed embeddings."""
        
        scores = {}
        explanations = {}
        
        # 1. Semantic similarity (ENHANCED with batch optimization)
        scores['semantic'], explanations['semantic'] = self.compute_semantic_similarity(
            req_doc, act_doc, req_idx, act_idx
        )
        
        # 2-4. All other components unchanged (already fast)
        scores['bm25'], explanations['bm25'] = self.compute_bm25_score(req_terms, act_terms, corpus_stats)
        scores['domain'], explanations['domain'] = self.compute_domain_similarity(req_terms, act_terms, domain_weights)
        
        req_terms_for_expansion = [token.lemma_.lower() for token in req_doc 
                                if (token.pos_ in ['NOUN', 'VERB', 'ADJ'] and 
                                    not token.is_stop and len(token.text) > 2)]
        scores['query_expansion'], explanations['query_expansion'] = self.expand_query_aerospace(
            req_terms_for_expansion, act_terms
        )
                
        return scores, explanations
    
    def create_match_explanation(self, req_id: str, req_text: str, act_name: str,
                               scores: Dict[str, float], explanations: Dict[str, str],
                               weights: Dict[str, float], req_terms: List[str], 
                               act_terms: List[str]) -> MatchExplanation:
        """Create detailed match explanation."""
        
        # Calculate combined score
        combined_score = sum(weights.get(key, 0) * score for key, score in scores.items())
        
        # Determine match quality
        if combined_score >= 0.6:
            match_quality = "EXCELLENT"
        elif combined_score >= 0.45:
            match_quality = "GOOD"
        elif combined_score >= 0.3:
            match_quality = "MODERATE"
        else:
            match_quality = "WEAK"
        
        # Semantic similarity level
        semantic_score = scores.get('semantic', 0)
        if semantic_score >= 0.7:
            semantic_level = "Very High"
        elif semantic_score >= 0.5:
            semantic_level = "High"
        elif semantic_score >= 0.3:
            semantic_level = "Medium"
        else:
            semantic_level = "Low"
        
        # Find meaningful shared terms (prioritize aerospace)
        shared_terms = list(set(req_terms) & set(act_terms))
        aerospace_shared = [t for t in shared_terms if t in self.all_aerospace_terms]
        other_shared = [t for t in shared_terms if t not in self.all_aerospace_terms and len(t) > 3]
        meaningful_shared = aerospace_shared[:3] + other_shared[:2]
        
        return MatchExplanation(
            requirement_id=req_id,
            requirement_text=req_text[:100] + "..." if len(req_text) > 100 else req_text,
            activity_name=act_name,
            combined_score=combined_score,
            semantic_score=scores.get('semantic', 0),
            bm25_score=scores.get('bm25', 0),
            syntactic_score=0.0,  # Simplified - removed syntactic for clarity
            domain_score=scores.get('domain', 0),
            query_expansion_score=scores.get('query_expansion', 0),
            semantic_explanation=explanations.get('semantic', 'N/A'),
            bm25_explanation=explanations.get('bm25', 'N/A'),
            syntactic_explanation='N/A',  # Simplified
            domain_explanation=explanations.get('domain', 'N/A'),
            query_expansion_explanation=explanations.get('query_expansion', 'N/A'),
            shared_terms=meaningful_shared,
            semantic_similarity_level=semantic_level,
            match_quality=match_quality
        )
    
    def run_matching(self, requirements_file: str = "requirements.csv",
                    activities_file: str = "activities.csv",
                    weights: Optional[Dict[str, float]] = None,
                    min_similarity: float = 0.15,
                    top_n: int = 5,
                    output_file: str = "aerospace_matches",
                    save_explanations: bool = True) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Run aerospace-optimized matching with batched embeddings.
        
        Returns:
            Tuple of (matches_dataframe, run_parameters)
        """
        
        # Default aerospace-optimized weights
        if weights is None:
            weights = {
                'semantic': 1,        # Moderate - general models struggle with aerospace
                'bm25': 1,           # High - term matching crucial in technical domains
                'domain': 1,         # High - aerospace terms are key
                'query_expansion': 1 # Moderate - helps with sparse activities 
            }
            logger.info("🚀 Using aerospace-optimized weights")
        
        # Resolve file paths
        file_mapping = {
            'requirements': requirements_file,
            'activities': activities_file
        }
        resolved_paths = self.path_resolver.resolve_input_files(file_mapping)
        requirements_file = resolved_paths['requirements']
        activities_file = resolved_paths['activities']
        
        # Load data
        logger.info("📂 Loading input files...")
        # requirements_df = self.file_handler.safe_read_csv(requirements_file).fillna({"Requirement Text": ""})
        # activities_df = self.file_handler.safe_read_csv(activities_file).fillna({"Activity Name": ""})
        
        requirements_df = self.load_requirements(resolved_paths['requirements'])
        activities_df = self.load_activities(resolved_paths['activities'])

        logger.info(f"📊 Loaded {len(requirements_df)} requirements and {len(activities_df)} activities")
        
        # OPTIMIZATION: Precompute embeddings once for massive speedup
        req_texts = list(requirements_df["Requirement Text"])
        act_texts = list(activities_df["Activity Name"])
        
        
        # Prepare corpus for analysis
        req_texts = list(requirements_df["Requirement Text"])
        act_texts = list(activities_df["Activity Name"])
        all_texts = [text for text in req_texts + act_texts if text.strip()]
        
        # Extract domain weights
        logger.info("🛸 Extracting aerospace domain terms...")
        domain_weights = self.extract_domain_weights(all_texts)
        
        # Compute corpus statistics
        logger.info("📈 Computing corpus statistics...")
        doc_freq = Counter()
        all_terms = []
        
        for text in all_texts:
            terms = self._preprocess_text_aerospace(text)
            all_terms.extend(terms)
            for term in set(terms):
                doc_freq[term] += 1
        
        corpus_stats = {
            'total_docs': len(all_texts),
            'avg_doc_length': len(all_terms) / len(all_texts) if all_texts else 0,
            'doc_freq': doc_freq
        }
        
        # Process requirements and find matches
        logger.info("🔍 Processing matches with batch optimization...")
        matches = []
        explanations = []
        
        for req_idx, req_row in requirements_df.iterrows():
            req_id = req_row.get("ID", f"REQ_{req_idx}")
            req_text = req_row["Requirement Text"]
            
            if not req_text.strip():
                continue
            
            # Process requirement (unchanged)
            req_doc = self.nlp(req_text)
            req_terms = self._preprocess_text_aerospace(req_text)
            
            activity_scores = []
            
            for act_idx, act_row in activities_df.iterrows():
                act_name = act_row["Activity Name"]
                
                if not act_name.strip():
                    continue
                
                # Process activity (unchanged)
                act_doc = self.nlp(act_name)
                act_terms = self._preprocess_text_aerospace(act_name)
                
                # OPTIMIZED: Pass indices for batch embedding lookup
                scores, score_explanations = self.compute_comprehensive_similarity(
                    req_doc, act_doc, req_terms, act_terms, corpus_stats, domain_weights,
                    req_idx=req_idx, act_idx=act_idx  # NEW: Pass indices for optimization
                )
                
                # Calculate combined score (unchanged)
                combined_score = sum(weights[key] * score for key, score in scores.items()) / sum(weights.values())
                
                if combined_score >= min_similarity:
                    activity_scores.append({
                        'activity_idx': act_idx,
                        'activity_name': act_name,
                        'combined_score': combined_score,
                        'scores': scores,
                        'explanations': score_explanations
                    })
            
            # Sort and take top N (unchanged)
            activity_scores.sort(key=lambda x: x['combined_score'], reverse=True)
            top_matches = activity_scores[:top_n]
            
            # Create match records (unchanged)
            for match in top_matches:
                matches.append({
                    'Requirement_ID': req_id,
                    'Requirement_Text': req_text,
                    'Activity_Name': match['activity_name'],
                    'Combined_Score': match['combined_score'],
                    'Semantic_Score': match['scores'].get('semantic', 0),
                    'BM25_Score': match['scores'].get('bm25', 0),
                    'Domain_Score': match['scores'].get('domain', 0),
                    'Query_Expansion_Score': match['scores'].get('query_expansion', 0)
                })
                
                # Create detailed explanation (unchanged)
                if save_explanations:
                    explanation = self.create_match_explanation(
                        req_id, req_text, match['activity_name'],
                        match['scores'], match['explanations'],
                        weights, req_terms, 
                        self._preprocess_text_aerospace(match['activity_name'])
                    )
                    explanations.append(explanation)
        
        # Create results DataFrame
        if matches:
            matches_df = pd.DataFrame(matches)
            
            # Save results
            results_dir = self.repo_manager.get_results_path()
            results_file = results_dir / f"{output_file}.csv"
            matches_df.to_csv(results_file, index=False)
            logger.info(f"💾 Results saved to: {results_file}")
            
            # Save explanations
            if save_explanations and explanations:
                explanations_file = results_dir / f"{output_file}_explanations.json"
                explanations_data = []
                
                for exp in explanations:
                    explanations_data.append({
                        'requirement_id': exp.requirement_id,
                        'requirement_text': exp.requirement_text,
                        'activity_name': exp.activity_name,
                        'combined_score': exp.combined_score,
                        'match_quality': exp.match_quality,
                        'scores': {
                            'semantic': exp.semantic_score,
                            'bm25': exp.bm25_score,
                            'domain': exp.domain_score,
                            'query_expansion': exp.query_expansion_score
                        },
                        'explanations': {
                            'semantic': exp.semantic_explanation,
                            'bm25': exp.bm25_explanation,
                            'domain': exp.domain_explanation,
                            'query_expansion': exp.query_expansion_explanation
                        },
                        'shared_terms': exp.shared_terms,
                        'semantic_similarity_level': exp.semantic_similarity_level
                    })
                
                with open(explanations_file, 'w', encoding='utf-8') as f:
                    json.dump(explanations_data, f, indent=2, ensure_ascii=False)
                
                logger.info(f"📄 Explanations saved to: {explanations_file}")
            
            # Print analysis
            self._print_matching_analysis(matches_df, weights)
            
        else:
            logger.warning("⚠️ No matches found with current parameters")
            matches_df = pd.DataFrame()
        
        run_parameters = {
            'weights': weights,
            'min_similarity': min_similarity,
            'top_n': top_n,
            'output_file': output_file,
            'requirements_file': requirements_file,
            'activities_file': activities_file,
            'save_explanations': save_explanations
        }
        
        return matches_df, run_parameters
    
    def _print_matching_analysis(self, matches_df: pd.DataFrame, weights: Dict[str, float]):
        """Print clean matching analysis."""
        
        print(f"\n{'='*70}")
        print("🚀 AEROSPACE MATCHER ANALYSIS")
        print(f"{'='*70}")
        
        # Basic statistics
        print(f"\n📊 Match Statistics:")
        print(f"  Total matches: {len(matches_df)}")
        print(f"  Requirements matched: {len(matches_df['Requirement_ID'].unique())}")
        print(f"  Average matches per requirement: {len(matches_df) / len(matches_df['Requirement_ID'].unique()):.1f}")
        
        # Score distribution
        score_thresholds = {
            'excellent': 0.6,
            'good': 0.45,
            'moderate': 0.3,
            'weak': 0.15
        }
        
        print(f"\n🎯 Match Quality Distribution:")
        for quality, threshold in score_thresholds.items():
            count = len(matches_df[matches_df['Combined_Score'] >= threshold])
            percentage = count / len(matches_df) * 100 if len(matches_df) > 0 else 0
            print(f"  {quality.capitalize()} (≥{threshold}): {count} ({percentage:.1f}%)")
        
        # Component analysis
        print(f"\n⚙️ Component Performance:")
        components = ['Semantic_Score', 'BM25_Score', 'Domain_Score', 'Query_Expansion_Score']
        for comp in components:
            if comp in matches_df.columns:
                avg_score = matches_df[comp].mean()
                weight_key = comp.lower().replace('_score', '')
                weight = weights.get(weight_key, 0)
                contribution = avg_score * weight
                print(f"  {comp.replace('_', ' ')}: avg={avg_score:.3f}, weight={weight:.2f}, contribution={contribution:.3f}")
        
        # Aerospace term analysis
        aerospace_matches = 0
        for _, row in matches_df.iterrows():
            req_terms = set(self._preprocess_text_aerospace(row['Requirement_Text']))
            act_terms = set(self._preprocess_text_aerospace(row['Activity_Name']))
            shared_aerospace = (req_terms & act_terms) & self.all_aerospace_terms
            if shared_aerospace:
                aerospace_matches += 1
        
        print(f"\n🛸 Aerospace Domain Analysis:")
        print(f"  Matches with aerospace terms: {aerospace_matches}")
        if len(matches_df) > 0:
            print(f"  Aerospace coverage: {aerospace_matches/len(matches_df)*100:.1f}%")
        
        # Overall assessment
        avg_score = matches_df['Combined_Score'].mean() if len(matches_df) > 0 else 0
        excellent_pct = len(matches_df[matches_df['Combined_Score'] >= 0.6]) / len(matches_df) * 100 if len(matches_df) > 0 else 0
        
        print(f"\n🎯 Overall Assessment:")
        if avg_score >= 0.4 and excellent_pct > 10:
            assessment = "🚀 EXCELLENT: Strong aerospace matching performance"
        elif avg_score >= 0.35:
            assessment = "✅ GOOD: Solid aerospace matching"
        elif avg_score >= 0.3:
            assessment = "📈 MODERATE: Acceptable performance"
        else:
            assessment = "🔧 NEEDS TUNING: Consider adjusting parameters"
        
        print(f"  {assessment}")
        print(f"  Average score: {avg_score:.3f}")
        print(f"  High-quality matches: {excellent_pct:.1f}%")

def main():
    """Run aerospace matching with enhanced evaluation parameter passing."""
    print("="*70)
    print("🚀 AEROSPACE REQUIREMENTS MATCHER")
    print("="*70)
    
    # Check dependencies
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        print("⚠️ Install sentence-transformers for best performance: pip install sentence-transformers")
    if not SKLEARN_AVAILABLE:
        print("⚠️ Install scikit-learn for TF-IDF: pip install scikit-learn")
    
    # Setup repository
    try:
        repo_manager = RepositoryStructureManager("outputs")
        repo_manager.setup_repository_structure()
    except Exception as e:
        print(f"⚠️ Repository setup issue: {e}")
        import os
        os.makedirs('outputs/matching_results', exist_ok=True)
        repo_manager = RepositoryStructureManager("outputs")
    
    # Create matcher
    matcher = AerospaceMatcher(repo_manager=repo_manager)
    
    try:
        # Run matching and GET the parameters it returns
        results_df, run_parameters = matcher.run_matching(
            requirements_file="requirements.csv",
            activities_file="activities.csv",
            min_similarity=0.15,
            top_n=5,
            output_file="aerospace_matches",
            save_explanations=True
        )
        
        print(f"\n✅ Aerospace matching complete!")
        print(f"📁 Results: outputs/matching_results/aerospace_matches.csv")
        print(f"📄 Explanations: outputs/matching_results/aerospace_matches_explanations.json")
        print(f"📊 Total matches: {len(results_df)}")
        
        # === ENHANCED EVALUATION WITH ACTUAL RUN PARAMETERS ===
        print(f"\n📊 Running enhanced evaluation with actual run configuration...")
        eval_results = None
        try:
            from src.evaluation.simple_evaluation import FixedSimpleEvaluator
            
            # Add matcher-specific information to the run_parameters
            run_parameters.update({
                'spacy_model': getattr(matcher.nlp, 'meta', {}).get('name', 'unknown') if hasattr(matcher, 'nlp') else 'unknown',
                'matcher_class': matcher.__class__.__name__,
                'aerospace_terms_count': len(matcher.all_aerospace_terms) if hasattr(matcher, 'all_aerospace_terms') else 0,
                'synonyms_count': len(matcher.synonyms) if hasattr(matcher, 'synonyms') else 0,
            })
            
            evaluator = FixedSimpleEvaluator()
            eval_results = evaluator.evaluate_matches(
                matches_file="outputs/matching_results/aerospace_matches.csv",
                ground_truth_file="manual_matches.csv",
                requirements_file="requirements.csv",
                run_params=run_parameters  # Use the ACTUAL parameters from run_matching!
            )
            
            if "error" not in eval_results:
                print(f"✅ Enhanced evaluation complete!")
                print(f"📊 F1@5: {eval_results['metrics']['f1_at_5']:.3f}")
                print(f"📈 Coverage: {eval_results['metrics']['coverage']:.1%}")
                print(f"🎯 Perfect matches: {eval_results['metrics']['perfect_matches']}/{eval_results['metrics']['total_evaluated']}")
                print(f"🎯 Hit@5: {eval_results['metrics']['hit_at_5']:.1%}")
                
                # Show the configuration that was captured
                metadata = eval_results.get('metadata', {})
                if 'algorithm_parameters' in metadata:
                    params = metadata['algorithm_parameters']
                    print(f"\n🔧 CONFIGURATION CONFIRMED:")
                    print(f"   Threshold: {params.get('min_similarity_threshold', 'Unknown')}")
                    print(f"   Top-N: {params.get('top_n_matches', 'Unknown')}")
                    print(f"   Model: {run_parameters.get('spacy_model', 'Unknown')}")
                    
                if 'score_weights' in metadata:
                    weights = metadata['score_weights']
                    weights_str = ', '.join([f"{k}:{v}" for k, v in weights.items()])
                    print(f"   Weights: {weights_str}")
                
            else:
                print(f"⚠️ Evaluation failed: {eval_results['error']}")
                eval_results = None
        
        except ImportError as e:
            print(f"⚠️ Enhanced evaluator not available: {e}")
        except Exception as e:
            print(f"⚠️ Evaluation error: {e}")
            import traceback
            traceback.print_exc()

        # === USE EXISTING MatchingWorkbookGenerator ===
        print(f"\n📊 Creating matching workbook...")
        try:
            # Import existing workbook generator 
            from src.utils.matching_workbook_generator import MatchingWorkbookGenerator
            
            # Load requirements for context (using existing file handler)
            requirements_df = matcher.file_handler.safe_read_csv(
                matcher.path_resolver.resolve_input_files({'requirements': 'requirements.csv'})['requirements']
            )
            
            # Convert simple evaluation results to format expected by existing generator
            formatted_eval_results = None
            if eval_results and 'metrics' in eval_results:
                # Format simple evaluation results for existing MatchingWorkbookGenerator
                formatted_eval_results = {
                    'aggregate_metrics': {
                        'f1_at_5': {'mean': eval_results['metrics']['f1_at_5']},
                        'precision_at_5': {'mean': eval_results['metrics'].get('precision_at_5', 0)},
                        'recall_at_5': {'mean': eval_results['metrics'].get('recall_at_5', 0)}
                    },
                    'coverage': eval_results['metrics'].get('coverage', 0),
                    'total_requirements': eval_results['metrics'].get('total_evaluated', 0),
                    'covered_requirements': eval_results['metrics'].get('total_evaluated', 0)
                }
            
            # Use existing MatchingWorkbookGenerator.create_workbook method (UNCHANGED SIGNATURE)
            workbook_generator = MatchingWorkbookGenerator(repo_manager=repo_manager)
            workbook_path = workbook_generator.create_workbook(
                enhanced_df=results_df,  # Use as-is from matcher
                evaluation_results=formatted_eval_results,  # Convert simple eval results
                output_path=None,  # Use default path
                repo_manager=repo_manager
            )
            
            print(f"✅ Matching workbook created: {workbook_path}")
            
        except ImportError as e:
            print(f"⚠️ Workbook generator not available: {e}")
            workbook_path = None
        except Exception as e:
            print(f"⚠️ Workbook creation failed: {e}")
            workbook_path = None
        
        # === FINAL SUMMARY (unchanged) ===
        print(f"\n🎊 Generated Outputs:")
        print(f"   1. 📄 CSV results: outputs/matching_results/aerospace_matches.csv")
        print(f"   2. 📝 Match explanations: outputs/matching_results/aerospace_matches_explanations.json")
        
        if eval_results:
            print(f"   3. 📊 Evaluation report: outputs/evaluation_results/fixed_simple_evaluation_report.txt")
            print(f"   4. 📈 Evaluation metrics: outputs/evaluation_results/fixed_simple_metrics.json")
        
        if workbook_path:
            print(f"   5. 📋 Matching workbook: {workbook_path}")
        
        # Performance summary (unchanged)
        print(f"\n📈 Performance Summary:")
        avg_score = results_df['Combined_Score'].mean() if 'Combined_Score' in results_df.columns else 0
        print(f"   Average match score: {avg_score:.3f}")
        
        high_conf = len(results_df[results_df['Combined_Score'] >= 0.8]) if 'Combined_Score' in results_df.columns else 0
        print(f"   High confidence (≥0.8): {high_conf}")
        
        if eval_results and 'metrics' in eval_results:
            print(f"   F1@5 performance: {eval_results['metrics'].get('f1_at_5', 0):.3f}")
        
        print(f"\n✅ Workflow complete using existing components!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        
if __name__ == "__main__":
    main()