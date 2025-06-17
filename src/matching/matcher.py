"""
Enhanced Requirements Matcher - Aerospace Domain Optimization
Optimized for aerospace requirements matching with improved F1 scores
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
import logging
from pathlib import Path
from collections import Counter
import math
import spacy
import json
import re
from dataclasses import dataclass

# Performance improvement imports
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.feature_extraction.text import TfidfVectorizer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
    SKLEARN_AVAILABLE = True
except ImportError:
    print("⚠️ Missing dependencies. Install with: pip install sentence-transformers scikit-learn")
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    SKLEARN_AVAILABLE = False

# Import existing utils
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

try:
    from src.utils.file_utils import SafeFileHandler
    from src.utils.path_resolver import SmartPathResolver
    from src.utils.repository_setup import RepositoryStructureManager
except ImportError:
    print("⚠️ Utils not found - using fallback implementations")
    
    class SafeFileHandler:
        def __init__(self, repo_manager=None):
            self.repo_manager = repo_manager
        
        def safe_read_csv(self, file_path, **kwargs):
            return pd.read_csv(file_path, **kwargs)
        
        def get_structured_path(self, file_type, filename):
            if file_type == 'matching_results':
                os.makedirs('outputs/matching_results', exist_ok=True)
                return f'outputs/matching_results/{filename}'
            return filename
    
    class SmartPathResolver:
        def __init__(self, repo_manager=None):
            pass
        
        def resolve_input_files(self, file_mapping):
            return file_mapping
    
    class RepositoryStructureManager:
        def __init__(self, base_dir):
            self.base_dir = base_dir
            self.structure = {'matching_results': Path('outputs/matching_results')}
        
        def setup_repository_structure(self):
            os.makedirs('outputs/matching_results', exist_ok=True)
            return {}

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
    
    # Aerospace domain configuration
    AEROSPACE_TERMS = {
        'systems': ['avionics', 'telemetry', 'payload', 'spacecraft', 'satellite', 'launch', 'orbit', 
                    'subsystem', 'hardware', 'software', 'firmware', 'interface'],
        'operations': ['command', 'control', 'uplink', 'downlink', 'transmission', 'reception',
                       'communication', 'monitoring', 'tracking', 'navigation'],
        'components': ['antenna', 'transponder', 'sensor', 'actuator', 'thruster', 'gyroscope',
                       'transmitter', 'receiver', 'processor', 'computer'],
        'data': ['packet', 'frame', 'protocol', 'interface', 'bus', 'link', 'signal',
                 'message', 'telemetry', 'data', 'information'],
        'ground': ['ground', 'station', 'mission', 'segment', 'facility', 'control center',
                   'operations', 'terminal', 'earth'],
        'flight': ['flight', 'trajectory', 'attitude', 'guidance', 'navigation', 'control',
                   'propulsion', 'maneuver', 'stabilization'],
        'power': ['power', 'battery', 'solar', 'panel', 'array', 'voltage', 'current',
                  'electrical', 'energy', 'charge'],
        'thermal': ['thermal', 'temperature', 'heat', 'cooling', 'radiator', 'insulation',
                    'dissipation', 'management'],
        'requirements': ['shall', 'should', 'must', 'will', 'may', 'requirement', 'specification',
                         'performance', 'capability', 'function', 'constraint']
    }
    
    AEROSPACE_ABBREVIATIONS = {
        'cmd': 'command',
        'ctrl': 'control',
        'tx': 'transmit',
        'rx': 'receive',
        'gnd': 'ground',
        'gcs': 'ground control station',
        'sat': 'satellite',
        'tlm': 'telemetry',
        'gc': 'ground control',
        'rf': 'radio frequency',
        'comm': 'communication',
        'nav': 'navigation',
        'gnc': 'guidance navigation control',
        'adcs': 'attitude determination control system',
        'eps': 'electrical power system',
        'tcs': 'thermal control system',
        'cdh': 'command data handling',
        'fdir': 'fault detection isolation recovery',
        'leo': 'low earth orbit',
        'geo': 'geostationary orbit',
        'meo': 'medium earth orbit',
        'aos': 'acquisition of signal',
        'los': 'loss of signal',
        'tle': 'two line element',
        's/c': 'spacecraft',
        'g/s': 'ground station'
    }
    AEROSPACE_RELATIONSHIPS = {
    'complementary_pairs': [
        ('transmit', 'receive'),
        ('uplink', 'downlink'),
        ('command', 'telemetry'),
        ('send', 'acquire'),
        ('broadcast', 'reception'),
        ('output', 'input'),
        ('encode', 'decode'),
        ('compress', 'decompress'),
        ('encrypt', 'decrypt')
    ],
    'workflow_sequences': [
        ['receive', 'process', 'store'],
        ['command', 'execute', 'verify'],
        ['acquire', 'track', 'communicate'],
        ['uplink', 'process', 'downlink'],
        ['sense', 'compute', 'actuate']
    ],
    'system_interactions': {
        'ground': ['spacecraft', 'satellite', 'vehicle'],
        'uplink': ['spacecraft', 'payload', 'transponder'],
        'command': ['control', 'execution', 'verification'],
        'telemetry': ['monitoring', 'analysis', 'storage']
    }
}
    
    def __init__(self, model_name: str = "en_core_web_trf", repo_manager=None):
        """Initialize aerospace-optimized matcher."""
        # Initialize spaCy
        try:
            self.nlp = spacy.load(model_name)
            logger.info(f"Loaded spaCy model: {model_name}")
        except OSError:
            logger.error(f"Could not load model {model_name}")
            raise
        
        # Store repository manager
        if repo_manager is None:
            raise ValueError("Repository manager is required")
        self.repo_manager = repo_manager
        
        # Initialize utils
        self.file_handler = SafeFileHandler(repo_manager)
        self.path_resolver = SmartPathResolver(repo_manager)
        
        # Initialize sentence transformers with aerospace-appropriate model
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                # Try scientific model first (better for technical text)
                try:
                    self.semantic_model = SentenceTransformer('allenai-specter')
                    logger.info("✅ Loaded scientific sentence transformer model (allenai-specter)")
                except:
                    # Fallback to general model
                    self.semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
                    logger.info("✅ Loaded general sentence transformer model (all-MiniLM-L6-v2)")
                
                # Pre-warm the model
                _ = self.semantic_model.encode(["test"], show_progress_bar=False)
                self.use_enhanced_semantic = True
                self.semantic_cache = {}
            except Exception as e:
                logger.warning(f"⚠️ Could not load Sentence Transformers: {e}")
                self.use_enhanced_semantic = False
        else:
            self.use_enhanced_semantic = False
        
        # Load aerospace synonyms
        self.synonyms = self._load_aerospace_synonyms()
        
        # Caches
        self.preprocessing_cache = {}
        self.expansion_cache = {}
        self.domain_embeddings = None
        
        # Create flattened aerospace terms set for quick lookup
        self.all_aerospace_terms = set()
        for category, terms in self.AEROSPACE_TERMS.items():
            self.all_aerospace_terms.update(terms)
    
    def _load_aerospace_synonyms(self) -> Dict[str, List[str]]:
        """Load aerospace-specific synonyms."""
        # Try to load from file first
        try:
            with open("synonyms.json", 'r') as f:
                base_synonyms = json.load(f)
                logger.info(f"Loaded base synonym dictionary with {len(base_synonyms)} entries")
        except FileNotFoundError:
            base_synonyms = {}
        
        # Add aerospace-specific synonyms
        aerospace_synonyms = {
            "command": ["control", "instruction", "directive", "order", "cmd"],
            "transmit": ["send", "broadcast", "uplink", "relay", "tx", "transmission"],
            "receive": ["reception", "acquire", "downlink", "obtain", "rx", "acquisition"],
            "ground": ["earth", "terrestrial", "base", "station", "gnd"],
            "data": ["information", "telemetry", "packet", "message", "signal"],
            "satellite": ["spacecraft", "vehicle", "platform", "asset", "s/c"],
            "control": ["manage", "regulate", "command", "operate", "monitor"],
            "system": ["subsystem", "component", "unit", "module", "assembly"],
            "interface": ["connection", "link", "port", "protocol", "bus"],
            "requirement": ["specification", "constraint", "criteria", "condition"],
            "performance": ["capability", "capacity", "efficiency", "operation"],
            "mission": ["operation", "task", "objective", "goal", "purpose"],
            "attitude": ["orientation", "position", "pointing", "stabilization"],
            "orbit": ["trajectory", "path", "track", "orbital"],
            "communication": ["comm", "comms", "link", "transmission", "signal"],
            "navigation": ["nav", "guidance", "positioning", "tracking"],
            "payload": ["instrument", "equipment", "cargo", "package"],
            "power": ["electrical", "energy", "battery", "solar", "eps"],
            "thermal": ["temperature", "heat", "cooling", "tcs"],
            "fault": ["failure", "error", "anomaly", "malfunction", "fdir"]
        }
        
        # Merge with base synonyms
        for term, synonyms in aerospace_synonyms.items():
            if term in base_synonyms:
                base_synonyms[term].extend([s for s in synonyms if s not in base_synonyms[term]])
            else:
                base_synonyms[term] = synonyms
        
        return base_synonyms
    
    def _expand_aerospace_abbreviations(self, text: str) -> str:
        """Expand common aerospace abbreviations in text."""
        text_lower = text.lower()
        
        # Sort abbreviations by length (longest first) to avoid partial replacements
        sorted_abbrevs = sorted(self.AEROSPACE_ABBREVIATIONS.items(), 
                               key=lambda x: len(x[0]), reverse=True)
        
        for abbr, full in sorted_abbrevs:
            # Match abbreviation with word boundaries
            pattern = r'\b' + re.escape(abbr) + r'\b'
            text_lower = re.sub(pattern, full, text_lower)
        
        return text_lower
    
    def _cached_preprocessing(self, text: str) -> List[str]:
        """Aerospace-aware preprocessing with caching."""
        if text in self.preprocessing_cache:
            return self.preprocessing_cache[text]
        
        # Expand abbreviations first
        expanded_text = self._expand_aerospace_abbreviations(text)
        
        doc = self.nlp(expanded_text)
        terms = []
        
        for token in doc:
            # Standard terms with aerospace filtering
            if (not token.is_stop and 
                not token.is_punct and 
                len(token.text) > 1 and  # Allow 2-letter aerospace terms
                token.is_alpha):
                
                # Keep aerospace terms even if they might be filtered otherwise
                if token.lemma_.lower() in self.all_aerospace_terms:
                    terms.append(token.lemma_.lower())
                elif token.pos_ in ['NOUN', 'VERB', 'ADJ', 'PROPN']:
                    terms.append(token.lemma_.lower())
            
            # Technical patterns
            if re.match(r'\d+\w+', token.text):  # "5ms", "100MHz"
                terms.append(token.text.lower())
            
            # Aerospace acronyms (keep uppercase)
            if token.text.isupper() and len(token.text) >= 2:
                terms.append(token.text.lower())
        
        # Add aerospace-specific bigrams
        if len(terms) > 1:
            for i in range(len(terms) - 1):
                # Create bigrams for aerospace term combinations
                if (terms[i] in self.all_aerospace_terms or 
                    terms[i+1] in self.all_aerospace_terms):
                    bigram = f"{terms[i]}_{terms[i+1]}"
                    terms.append(bigram)
        
        # Cache the result
        self.preprocessing_cache[text] = terms
        return terms
    
    def extract_aerospace_domain_terms(self, corpus: List[str]) -> Dict[str, float]:
        """Extract aerospace-specific domain terms using TF-IDF."""
        if not SKLEARN_AVAILABLE:
            logger.warning("sklearn not available, using fallback domain extraction")
            return self._fallback_domain_extraction(corpus)
        
        try:
            # Preprocess corpus for aerospace
            processed_corpus = [self._expand_aerospace_abbreviations(text) for text in corpus]
            
            # Configure TF-IDF for aerospace domain
            vectorizer = TfidfVectorizer(
                max_features=1000,      # More features for technical domain
                ngram_range=(1, 3),     # Include up to trigrams
                min_df=2,               # Must appear in at least 2 documents
                max_df=0.8,             # Not in more than 80% of documents
                token_pattern=r'\b[a-zA-Z0-9][a-zA-Z0-9_\-\.\/]*[a-zA-Z0-9]\b',  # Include technical terms
                lowercase=True,
                stop_words='english'
            )
            
            tfidf_matrix = vectorizer.fit_transform(processed_corpus)
            feature_names = vectorizer.get_feature_names_out()
            
            # Get average TF-IDF scores
            avg_tfidf = tfidf_matrix.mean(axis=0).A1
            
            # Create initial weights
            domain_weights = {}
            
            for term, weight in zip(feature_names, avg_tfidf):
                # Filter out too short or too common terms
                if len(term) >= 2 and weight > 0.01:
                    # Boost aerospace terms
                    boost = 1.0
                    
                    # Check if term contains aerospace keywords
                    term_lower = term.lower()
                    if any(aero_term in term_lower for aero_term in self.all_aerospace_terms):
                        boost *= 2.0
                    
                    # Boost terms with numbers (often technical specs)
                    if any(char.isdigit() for char in term):
                        boost *= 1.3
                    
                    # Boost multi-word technical phrases
                    if '_' in term or len(term.split()) > 1:
                        boost *= 1.2
                    
                    domain_weights[term] = weight * boost
            
            # Normalize weights
            if domain_weights:
                max_weight = max(domain_weights.values())
                domain_weights = {term: weight / max_weight 
                                 for term, weight in domain_weights.items()}
            
            logger.info(f"Extracted {len(domain_weights)} aerospace domain terms")
            
            # Log top aerospace terms
            top_terms = sorted(domain_weights.items(), key=lambda x: x[1], reverse=True)[:20]
            logger.info(f"Top aerospace terms: {[f'{t}({w:.3f})' for t, w in top_terms[:10]]}")
            
            return domain_weights
            
        except Exception as e:
            logger.error(f"TF-IDF extraction failed: {e}")
            return self._fallback_domain_extraction(corpus)
    
    def _fallback_domain_extraction(self, corpus: List[str]) -> Dict[str, float]:
        """Fallback domain extraction when sklearn is not available."""
        domain_weights = {}
        
        # Use aerospace terms with predefined weights
        for category, terms in self.AEROSPACE_TERMS.items():
            category_weight = {
                'requirements': 0.9,
                'operations': 0.8,
                'systems': 0.7,
                'components': 0.6,
                'data': 0.5,
                'ground': 0.5,
                'flight': 0.6,
                'power': 0.5,
                'thermal': 0.5
            }.get(category, 0.5)
            
            for term in terms:
                domain_weights[term] = category_weight
        
        return domain_weights
    
    def _is_technical_term(self, term: str) -> bool:
        """Check if a term is likely technical/aerospace."""
        term_lower = term.lower()
        
        # Direct aerospace term
        if term_lower in self.all_aerospace_terms:
            return True
        
        # Contains aerospace substring
        if any(aero in term_lower for aero in self.all_aerospace_terms):
            return True
        
        # Technical patterns
        if any([
            re.match(r'.*\d+.*', term),  # Contains numbers
            '_' in term or '-' in term,   # Technical separators
            term.isupper() and len(term) >= 2,  # Acronyms
            len(term) > 10  # Long technical terms
        ]):
            return True
        
        return False
    
    def compute_semantic_similarity_with_explanation(self, req_doc, act_doc):
        """Aerospace-optimized semantic similarity without aggressive calibration."""
        req_text = req_doc.text.strip()
        act_text = act_doc.text.strip()
        cache_key = f"{hash(req_text)}_{hash(act_text)}"
        
        if hasattr(self, 'semantic_cache') and cache_key in self.semantic_cache:
            return self.semantic_cache[cache_key]
        
        if self.use_enhanced_semantic:
            try:
                # Expand abbreviations for better semantic matching
                req_expanded = self._expand_aerospace_abbreviations(req_text)
                act_expanded = self._expand_aerospace_abbreviations(act_text)
                
                # Use sentence transformers
                embeddings = self.semantic_model.encode([req_expanded, act_expanded], 
                                                       show_progress_bar=False)
                similarity = float(cosine_similarity([embeddings[0]], [embeddings[1]])[0][0])
                
                # Aerospace-specific adjustments
                req_terms = set(req_expanded.lower().split())
                act_terms = set(act_expanded.lower().split())
                
                # Check for aerospace term overlap
                shared_aerospace = (req_terms & act_terms) & self.all_aerospace_terms
                
                # Boost similarity when aerospace terms match
                if shared_aerospace:
                    # Each shared aerospace term adds a small boost
                    boost = min(0.2, len(shared_aerospace) * 0.05)
                    similarity = min(1.0, similarity + boost)
                
                # Simple classification without calibration
                if similarity >= 0.7:
                    level = "Very High"
                elif similarity >= 0.5:
                    level = "High"
                elif similarity >= 0.35:
                    level = "Medium"
                elif similarity >= 0.2:
                    level = "Low"
                else:
                    level = "Very Low"
                
                explanation = f"{level} similarity ({similarity:.3f})"
                if shared_aerospace:
                    explanation += f" [+aerospace: {', '.join(list(shared_aerospace)[:3])}]"
                
                # Cache result
                if hasattr(self, 'semantic_cache'):
                    self.semantic_cache[cache_key] = (similarity, explanation)
                
                return similarity, explanation
                
            except Exception as e:
                logger.warning(f"Enhanced semantic similarity failed: {e}")
        
        # Fallback to spaCy similarity
        try:
            similarity = req_doc.similarity(act_doc)
            level = "High" if similarity >= 0.5 else "Medium" if similarity >= 0.3 else "Low"
            explanation = f"{level} similarity ({similarity:.3f}) via spaCy"
            return similarity, explanation
        except:
            return 0.0, "Similarity computation failed"
        
    def adjust_semantic_for_aerospace_relationships(self, req_text, act_text, base_similarity):
        """Adjust semantic similarity based on aerospace relationships."""
        
        req_words = set(req_text.lower().split())
        act_words = set(act_text.lower().split())
        
        # Check complementary pairs
        for word1, word2 in self.AEROSPACE_RELATIONSHIPS['complementary_pairs']:
            if (word1 in req_words and word2 in act_words) or \
            (word2 in req_words and word1 in act_words):
                # These are complementary in aerospace context
                adjustment = 0.3  # Significant boost
                return min(1.0, base_similarity + adjustment), "Aerospace complementary pair"
        
        # Check workflow relationships
        for workflow in self.AEROSPACE_RELATIONSHIPS['workflow_sequences']:
            req_matches = [w for w in workflow if any(w in word for word in req_words)]
            act_matches = [w for w in workflow if any(w in word for word in act_words)]
            
            if req_matches and act_matches:
                # Same workflow, different stages
                distance = abs(workflow.index(req_matches[0]) - workflow.index(act_matches[0]))
                adjustment = 0.2 / (distance + 1)  # Closer stages get higher boost
                return min(1.0, base_similarity + adjustment), f"Same workflow (distance: {distance})"
        
        return base_similarity, "No aerospace relationship found"
    
    def compute_bm25_score_with_explanation(self, query_terms: List[str], doc_terms: List[str], 
                                           corpus_stats: Dict[str, Any]) -> Tuple[float, str]:
        """BM25 scoring with proper normalization for aerospace domain."""
        score = 0.0
        doc_len = len(doc_terms)
        avgdl = corpus_stats.get('avg_doc_length', doc_len)
        N = corpus_stats.get('total_docs', 1)
        
        # BM25 parameters
        k1 = 1.2
        b = 0.75
        
        matching_terms = []
        term_scores = []
        aerospace_matches = []
        
        for term in set(query_terms):
            if term in doc_terms:
                tf = doc_terms.count(term)
                df = corpus_stats.get('doc_freq', {}).get(term, 1)
                
                # BM25 formula
                idf = math.log((N - df + 0.5) / (df + 0.5))
                tf_component = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (doc_len / avgdl)))
                term_score = idf * tf_component
                
                # Boost aerospace terms
                if term in self.all_aerospace_terms:
                    term_score *= 1.5
                    aerospace_matches.append(term)
                
                score += term_score
                term_scores.append((term, term_score))
                matching_terms.append(term)
        
        # Sort by contribution
        term_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Normalize by maximum possible score
        if query_terms:
            # Calculate theoretical maximum
            max_possible = 0.0
            for term in set(query_terms):
                df = 1  # Best case
                idf = math.log((N - df + 0.5) / (df + 0.5))
                max_tf = k1 + 1
                boost = 1.5 if term in self.all_aerospace_terms else 1.0
                max_possible += idf * max_tf * boost
            
            normalized_score = score / max_possible if max_possible > 0 else 0
        else:
            normalized_score = 0
        
        # Create explanation
        if term_scores:
            top_terms = term_scores[:3]
            term_parts = [f"'{term}'({score:.2f})" for term, score in top_terms]
            explanation = f"Matched {len(matching_terms)} terms: {'; '.join(term_parts)}"
            if aerospace_matches:
                explanation += f" [aerospace: {', '.join(aerospace_matches[:2])}]"
        else:
            explanation = "No term matches"
        
        return normalized_score, explanation
    
    def extract_syntactic_features(self, doc: spacy.tokens.Doc) -> Dict[str, List]:
        """Extract syntactic features relevant to aerospace requirements."""
        features = {
            'dep_patterns': [],
            'pos_sequence': [],
            'entity_types': [],
            'verb_frames': [],
            'requirement_patterns': []  # New: aerospace requirement patterns
        }
        
        # Focus on meaningful patterns
        for token in doc:
            if (not token.is_stop and token.is_alpha and 
                token.pos_ in ['NOUN', 'VERB', 'ADJ', 'PROPN']):
                
                pattern = f"{token.dep_}:{token.pos_}"
                features['dep_patterns'].append(pattern)
                
                # Verb frames (especially modal verbs for requirements)
                if token.pos_ == "VERB":
                    if token.lemma_ in ['shall', 'should', 'must', 'will', 'may']:
                        features['requirement_patterns'].append(f"MODAL:{token.lemma_}")
                    
                    children = [child.dep_ for child in token.children 
                               if child.pos_ in ['NOUN', 'PROPN']]
                    if children:
                        features['verb_frames'].append(f"{token.lemma_}:{':'.join(sorted(children))}")
        
        # POS sequences
        meaningful_pos = [token.pos_ for token in doc 
                         if token.pos_ in ['NOUN', 'VERB', 'ADJ', 'PROPN'] and not token.is_space]
        
        for i in range(len(meaningful_pos) - 1):
            features['pos_sequence'].append(f"{meaningful_pos[i]}_{meaningful_pos[i+1]}")
        
        # Entity types
        features['entity_types'] = [ent.label_ for ent in doc.ents]
        
        return features
    
    def compute_syntactic_similarity_with_explanation(self, features1: Dict, features2: Dict) -> Tuple[float, str]:
        """Compute syntactic similarity with aerospace-aware weighting."""
        total_sim = 0.0
        weights = {
            'dep_patterns': 0.3,
            'pos_sequence': 0.2,
            'entity_types': 0.1,
            'verb_frames': 0.2,
            'requirement_patterns': 0.2  # Important for requirements matching
        }
        
        explanations = []
        
        for feature_type, weight in weights.items():
            set1 = set(features1.get(feature_type, []))
            set2 = set(features2.get(feature_type, []))
            
            if not set1 and not set2:
                jaccard = 1.0
            elif not set1 or not set2:
                jaccard = 0.0
            else:
                intersection = len(set1 & set2)
                union = len(set1 | set2)
                jaccard = intersection / union if union > 0 else 0.0
            
            total_sim += weight * jaccard
            
            if jaccard > 0.3 and len(explanations) < 2:
                shared = list(set1 & set2)[:2]
                explanations.append(f"{feature_type}: {jaccard:.2f}")
                if shared:
                    explanations[-1] += f" ({', '.join(shared)})"
        
        explanation = "; ".join(explanations) if explanations else "Limited syntactic overlap"
        return total_sim, explanation
    
    def compute_domain_similarity_with_explanation(self, req_terms: List[str], act_terms: List[str],
                                                 domain_weights: Dict[str, float]) -> Tuple[float, str]:
        """Compute domain similarity with aerospace term emphasis."""
        req_domain_terms = [term for term in req_terms if term in domain_weights]
        act_domain_terms = [term for term in act_terms if term in domain_weights]
        
        if not req_domain_terms or not act_domain_terms:
            return 0.0, "No domain terms found"
        
        common_domain = set(req_domain_terms) & set(act_domain_terms)
        if not common_domain:
            return 0.0, "No shared domain terms"
        
        # Weight by term importance and aerospace relevance
        domain_score = 0.0
        aerospace_bonus = 0.0
        
        for term in common_domain:
            term_weight = domain_weights[term]
            
            # Extra weight for aerospace terms
            if term in self.all_aerospace_terms:
                term_weight *= 1.5
                aerospace_bonus += 0.1
            
            domain_score += term_weight
        
        # Normalize
        normalization_factor = math.sqrt(len(req_domain_terms) * len(act_domain_terms))
        domain_score = (domain_score + aerospace_bonus) / normalization_factor
        
        # Cap at 1.0
        domain_score = min(1.0, domain_score)
        
        # Create explanation
        shared_with_weights = [(term, domain_weights[term]) for term in common_domain]
        shared_with_weights.sort(key=lambda x: x[1], reverse=True)
        top_shared = shared_with_weights[:3]
        
        explanation = f"Shared {len(common_domain)} domain terms: "
        explanation += ", ".join([f"'{term}'({weight:.2f})" for term, weight in top_shared])
        
        aerospace_shared = common_domain & self.all_aerospace_terms
        if aerospace_shared:
            explanation += f" [aerospace: {len(aerospace_shared)}]"
        
        return domain_score, explanation
    
    def expand_query_with_explanation(self, query_doc: spacy.tokens.Doc) -> Tuple[List[str], str]:
        """Query expansion using aerospace synonyms."""
        expanded_terms = []
        
        # Extract query terms
        query_terms = [token.lemma_.lower() for token in query_doc 
                      if (token.pos_ in ['NOUN', 'VERB', 'ADJ'] and 
                          not token.is_stop and
                          len(token.text) > 2 and
                          token.is_alpha)]
        
        # Use synonym dictionary
        for term in query_terms:
            if term in self.synonyms:
                # Add synonyms that aren't already in query
                for synonym in self.synonyms[term]:
                    if synonym not in query_terms and synonym not in expanded_terms:
                        expanded_terms.append(synonym)
        
        # Limit expansion
        expanded_terms = expanded_terms[:5]
        
        if expanded_terms:
            explanation = f"Expanded with {len(expanded_terms)} aerospace synonyms"
        else:
            explanation = "No expansion terms found"
        
        return expanded_terms, explanation
    
    def compute_comprehensive_similarity_with_explanation(self, req_doc: spacy.tokens.Doc, 
                                                        act_doc: spacy.tokens.Doc,
                                                        req_terms: List[str],
                                                        act_terms: List[str],
                                                        corpus_stats: Dict[str, Any],
                                                        domain_weights: Dict[str, float]) -> Tuple[Dict[str, float], Dict[str, str]]:
        """Compute all similarity components with aerospace optimizations."""
        scores = {}
        explanations = {}
        
        # 1. Semantic similarity (aerospace-aware)
        scores['dense_semantic'], explanations['semantic'] = self.compute_semantic_similarity_with_explanation(req_doc, act_doc)
        
        # 2. BM25 similarity (with aerospace term boosting)
        scores['bm25'], explanations['bm25'] = self.compute_bm25_score_with_explanation(req_terms, act_terms, corpus_stats)
        
        # 3. Syntactic similarity
        req_syntax = self.extract_syntactic_features(req_doc)
        act_syntax = self.extract_syntactic_features(act_doc)
        scores['syntactic'], explanations['syntactic'] = self.compute_syntactic_similarity_with_explanation(req_syntax, act_syntax)
        
        # 4. Domain similarity (aerospace-focused)
        scores['domain_weighted'], explanations['domain'] = self.compute_domain_similarity_with_explanation(req_terms, act_terms, domain_weights)
        
        # 5. Query expansion (aerospace synonyms)
        expanded_req, query_exp = self.expand_query_with_explanation(req_doc)
        if expanded_req:
            expansion_overlap = len(set(expanded_req) & set(act_terms))
            scores['query_expansion'] = expansion_overlap / len(expanded_req) if expanded_req else 0
            explanations['query_expansion'] = f"{expansion_overlap}/{len(expanded_req)} matches | {query_exp}"
        else:
            scores['query_expansion'] = 0.0
            explanations['query_expansion'] = "No expansion"
        
        return scores, explanations
    
    def create_match_explanation(self, req_id: str, req_text: str, act_name: str,
                               scores: Dict[str, float], explanations: Dict[str, str],
                               weights: Dict[str, float], req_terms: List[str], act_terms: List[str]) -> MatchExplanation:
        """Create detailed match explanation with aerospace context."""
        combined_score = sum(weights.get(key, 0) * score for key, score in scores.items())
        
        # Aerospace-aware quality thresholds
        if combined_score >= 0.6:
            match_quality = "EXCELLENT"
        elif combined_score >= 0.45:
            match_quality = "GOOD"
        elif combined_score >= 0.3:
            match_quality = "MODERATE"
        else:
            match_quality = "WEAK"
        
        # Semantic level
        semantic_score = scores.get('dense_semantic', 0)
        if semantic_score >= 0.7:
            semantic_level = "Very High"
        elif semantic_score >= 0.5:
            semantic_level = "High"
        elif semantic_score >= 0.3:
            semantic_level = "Medium"
        else:
            semantic_level = "Low"
        
        # Find shared terms, prioritizing aerospace terms
        shared_terms = list(set(req_terms) & set(act_terms))
        
        # Sort by aerospace relevance
        aerospace_shared = [t for t in shared_terms if t in self.all_aerospace_terms]
        other_shared = [t for t in shared_terms if t not in self.all_aerospace_terms and len(t) > 3]
        
        meaningful_shared = aerospace_shared[:3] + other_shared[:2]
        
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
            shared_terms=meaningful_shared,
            semantic_similarity_level=semantic_level,
            match_quality=match_quality
        )
    
    def run_final_matching(self, requirements_file: str = "requirements.csv",
                          activities_file: str = "activities.csv",
                          weights: Optional[Dict[str, float]] = None,
                          min_sim: float = 0.15,
                          top_n: int = 5,
                          out_file: str = "final_clean_matches",
                          save_explanations: bool = True,
                          repo_manager=None) -> pd.DataFrame:
        """
        Run aerospace-optimized matching process.
        """
        if repo_manager is not None:
            self.repo_manager = repo_manager
        
        # Aerospace-optimized weights
        if weights is None:
            weights = {
                'dense_semantic': 0.2,      # Lower - general models struggle with aerospace
                'bm25': 0.5,                # Higher - term matching crucial in technical domains
                'syntactic': 0.05,           # Minimal
                'domain_weighted': 0.25,     # Higher - aerospace terms are key
                'query_expansion': 0.0       # Disabled until properly tuned
            }
            logger.info("🚀 Using aerospace-optimized weight configuration")
        
        # Resolve file paths
        file_mapping = {
            'requirements': requirements_file,
            'activities': activities_file
        }
        
        resolved_paths = self.path_resolver.resolve_input_files(file_mapping)
        requirements_file = resolved_paths['requirements']
        activities_file = resolved_paths['activities']
        
        # Load data
        requirements_df = self.file_handler.safe_read_csv(requirements_file).fillna({"Requirement Text": ""})
        activities_df = self.file_handler.safe_read_csv(activities_file).fillna({"Activity Name": ""})
        
        logger.info(f"✅ Loaded {len(requirements_df)} requirements and {len(activities_df)} activities")
        
        # Prepare corpus
        req_texts = list(requirements_df["Requirement Text"])
        act_texts = list(activities_df["Activity Name"])
        all_texts = req_texts + act_texts
        all_texts = [text for text in all_texts if text.strip()]
        
        # Extract aerospace domain terms
        logger.info("🛸 Extracting aerospace domain terms...")
        domain_weights = self.extract_aerospace_domain_terms(all_texts)
        
        # Compute corpus statistics
        logger.info("📊 Computing corpus statistics...")
        all_term_lists = []
        doc_freq = Counter()
        
        for text in all_texts:
            terms = self._cached_preprocessing(text)
            all_term_lists.append(terms)
            doc_freq.update(set(terms))
        
        corpus_stats = {
            'total_docs': len(all_texts),
            'avg_doc_length': np.mean([len(terms) for terms in all_term_lists]),
            'doc_freq': dict(doc_freq)
        }
        
        # Process documents
        logger.info("🔧 Processing requirements and activities...")
        req_docs = list(self.nlp.pipe(req_texts, batch_size=32))
        act_docs = list(self.nlp.pipe(act_texts, batch_size=32))
        
        # Extract term lists
        req_term_lists = [self._cached_preprocessing(text) for text in req_texts]
        act_term_lists = [self._cached_preprocessing(text) for text in act_texts]
        
        # Matching phase
        logger.info("🎯 Running aerospace-optimized matching...")
        matches = []
        match_explanations = []
        total_reqs = len(requirements_df)
        
        # Statistics
        aerospace_match_count = 0
        high_score_count = 0
        
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
                
                # Compute similarity scores
                sim_scores, explanations = self.compute_comprehensive_similarity_with_explanation(
                    req_doc, act_doc, req_terms, act_terms, corpus_stats, domain_weights
                )
                
                # Calculate combined score
                combined_score = sum(weights[score_type] * score 
                                   for score_type, score in sim_scores.items())
                
                if combined_score >= min_sim:
                    # Track aerospace matches
                    shared_aerospace = set(req_terms) & set(act_terms) & self.all_aerospace_terms
                    if shared_aerospace:
                        aerospace_match_count += 1
                    
                    if combined_score >= 0.6:
                        high_score_count += 1
                    
                    # Get requirement ID
                    req_id = getattr(req_row, 'ID', None) or getattr(req_row, '_1', req_idx)
                    
                    # Create explanation if requested
                    if save_explanations:
                        explanation = self.create_match_explanation(
                            str(req_id), req_text, act_name, sim_scores, explanations, 
                            weights, req_terms, act_terms
                        )
                        match_explanations.append(explanation)
                    
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
        
        # Create results DataFrame
        matches_df = pd.DataFrame(matches)
        
        if not matches_df.empty:
            # Save results
            csv_file = self.file_handler.get_structured_path('matching_results', f"{out_file}.csv")
            Path(csv_file).parent.mkdir(parents=True, exist_ok=True)
            matches_df.to_csv(csv_file, index=False, encoding='utf-8')
            logger.info(f"✅ Saved {len(matches)} matches to {csv_file}")
            
            # Save explanations
            if save_explanations and match_explanations:
                explanations_file = self.file_handler.get_structured_path('matching_results', f"{out_file}_explanations.json")
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
                
                logger.info(f"✅ Saved explanations to {explanations_file}")
            
            # Print aerospace analysis
            self._print_aerospace_analysis(matches_df, weights, aerospace_match_count, high_score_count)
            
        else:
            logger.warning("No matches found with current parameters")
        
        return matches_df
    def analyze_ground_truth_terminology(self, ground_truth_file, requirements_file, activities_file):
        """Analyze terminology mismatches between ground truth and activities."""
        
        # Load files
        gt_df = pd.read_csv(ground_truth_file)
        req_df = pd.read_csv(requirements_file)
        act_df = pd.read_csv(activities_file)
        
        # Extract all activity names from ground truth
        gt_activities = []
        for _, row in gt_df.iterrows():
            if pd.notna(row['Satisfied By']):
                activities = [a.strip() for a in str(row['Satisfied By']).split(',')]
                gt_activities.extend(activities)
        
        # Compare with available activities
        available_activities = set(act_df['Activity Name'].str.lower())
        gt_activities_lower = [a.lower() for a in gt_activities]
        
        # Find mismatches
        not_found = [a for a in gt_activities if a.lower() not in available_activities]
        
        print(f"Ground truth activities not found in activities.csv: {len(not_found)}")
        print(f"Examples: {not_found[:5]}")
        
        # Check for partial matches
        partial_matches = []
        for gt_act in not_found:
            for avail_act in available_activities:
                if any(word in avail_act for word in gt_act.lower().split()):
                    partial_matches.append((gt_act, avail_act))
        
        print(f"\nPotential partial matches: {len(partial_matches)}")
        for gt, avail in partial_matches[:5]:
            print(f"  GT: '{gt}' might be '{avail}'")

    def _print_aerospace_analysis(self, matches_df: pd.DataFrame, weights: Dict[str, float],
                                 aerospace_match_count: int, high_score_count: int):
        """Print aerospace-specific matching analysis."""
        
        print(f"\n{'='*70}")
        print("🚀 AEROSPACE MATCHER ANALYSIS")
        print(f"{'='*70}")
        
        # Score thresholds
        thresholds = {
            'excellent': 0.6,
            'high': 0.45,
            'medium': 0.3,
            'low': 0.2
        }
        
        # Calculate metrics
        excellent = len(matches_df[matches_df['Combined Score'] >= thresholds['excellent']])
        high = len(matches_df[matches_df['Combined Score'] >= thresholds['high']])
        medium = len(matches_df[matches_df['Combined Score'] >= thresholds['medium']])
        
        print(f"\n📊 Match Quality Distribution:")
        print(f"  ✅ Excellent (≥{thresholds['excellent']}): {excellent} ({excellent/len(matches_df)*100:.1f}%)")
        print(f"  ✅ High (≥{thresholds['high']}): {high} ({high/len(matches_df)*100:.1f}%)")
        print(f"  ✅ Medium (≥{thresholds['medium']}): {medium} ({medium/len(matches_df)*100:.1f}%)")
        
        print(f"\n🛸 Aerospace Domain Analysis:")
        print(f"  Matches with aerospace terms: {aerospace_match_count}")
        print(f"  Aerospace term coverage: {aerospace_match_count/len(matches_df)*100:.1f}%")
        
        print(f"\n📈 Component Performance:")
        print(f"  Semantic avg: {matches_df['Dense Semantic'].mean():.3f}")
        print(f"  BM25 avg: {matches_df['BM25 Score'].mean():.3f}")
        print(f"  Domain avg: {matches_df['Domain Weighted'].mean():.3f}")
        
        print(f"\n⚡ Weight Configuration:")
        for component, weight in weights.items():
            if weight > 0:
                print(f"  {component}: {weight}")
        
        # Performance assessment
        avg_score = matches_df['Combined Score'].mean()
        if avg_score >= 0.4 and excellent/len(matches_df) > 0.1:
            assessment = "🚀 EXCELLENT: Strong aerospace matching performance"
        elif avg_score >= 0.35:
            assessment = "✅ GOOD: Solid aerospace matching"
        elif avg_score >= 0.3:
            assessment = "📈 MODERATE: Acceptable performance"
        else:
            assessment = "🔧 NEEDS TUNING: Consider adjusting parameters"
        
        print(f"\n{assessment}")
        print(f"\n💡 Tips: Check domain term extraction and consider domain-specific fine-tuning")


def main():
    """Run aerospace-optimized matcher."""
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
        os.makedirs('outputs/matching_results', exist_ok=True)
        repo_manager = RepositoryStructureManager("outputs")
    
    # Create matcher
    matcher = AerospaceMatcher(repo_manager=repo_manager)
    
    try:
        # Run matching
        results = matcher.run_final_matching(
            requirements_file="requirements.csv",
            activities_file="activities.csv",
            min_sim=0.15,
            top_n=5,
            out_file="aerospace_matches",
            save_explanations=True
        )
        
        print(f"\n✅ Aerospace matching complete!")
        print(f"📁 Results: outputs/matching_results/aerospace_matches.csv")
        print(f"📄 Explanations: outputs/matching_results/aerospace_matches_explanations.json")
        print(f"📊 Total matches: {len(results)}")
        
    except FileNotFoundError as e:
        print(f"\n❌ File not found: {e}")
        print(f"💡 Required files:")
        print(f"   • requirements.csv")
        print(f"   • activities.csv")
        print(f"   • synonyms.json (optional but recommended)")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()