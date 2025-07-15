"""
Domain Knowledge Builder - Learn from Existing Manual Traces
Extract patterns, vocabulary, and matching strategies from existing human-traced data
"""

import pandas as pd
import numpy as np
from collections import Counter, defaultdict
import spacy
import json
from typing import Dict, List, Tuple, Set
import logging
from pathlib import Path
import re
import sys
import os

# Add project root to path for imports
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent
sys.path.insert(0, str(project_root))

# Import project utilities
try:
    from src.utils.file_utils import SafeFileHandler
    from src.utils.path_resolver import SmartPathResolver
    from src.utils.repository_setup import RepositoryStructureManager
    UTILS_AVAILABLE = True
    print("✅ Successfully imported project utils")
except ImportError as e:
    print(f"⚠️ Could not import utils: {e}")
    UTILS_AVAILABLE = False

logger = logging.getLogger(__name__)

class DomainKnowledgeBuilder:
    """Extract domain knowledge from existing manual requirement-activity traces."""
    
    def __init__(self, spacy_model: str = "en_core_web_trf"):
        # Initialize spaCy with fallback models
        try:
            self.nlp = spacy.load(spacy_model)
            logger.info(f"Loaded spaCy model: {spacy_model}")
        except OSError:
            logger.warning(f"spaCy model '{spacy_model}' not found. Trying fallback models...")
            fallback_models = ["en_core_web_lg", "en_core_web_md", "en_core_web_sm"]
            model_loaded = False
            
            for fallback in fallback_models:
                try:
                    self.nlp = spacy.load(fallback)
                    logger.info(f"Loaded fallback spaCy model: {fallback}")
                    model_loaded = True
                    break
                except OSError:
                    continue
            
            if not model_loaded:
                logger.error("No spaCy models available. Using basic text processing.")
                self.nlp = None
        
        # Setup project utilities for smart file finding
        if UTILS_AVAILABLE:
            self.repo_manager = RepositoryStructureManager("outputs")
            self.repo_manager.setup_repository_structure()
            self.file_handler = SafeFileHandler(self.repo_manager)
            self.path_resolver = SmartPathResolver(self.repo_manager)
        else:
            self.repo_manager = None
            self.file_handler = None
            self.path_resolver = None
        
        # Knowledge storage containers
        self.domain_patterns = {}
        self.vocabulary_mappings = {}
        self.phrase_patterns = {}
        self.entity_mappings = {}
    
    def _find_manual_traces_file(self, filename: str) -> str:
        """Find manual traces file using project path resolution."""
        
        if UTILS_AVAILABLE and self.path_resolver:
            # Use project path resolution system
            file_mapping = {'ground_truth': filename}
            resolved_paths = self.path_resolver.resolve_input_files(file_mapping)
            resolved_path = resolved_paths['ground_truth']
            
            if Path(resolved_path).exists():
                print(f"✅ Found manual traces: {resolved_path}")
                return resolved_path
            else:
                print(f"❌ File not found at resolved path: {resolved_path}")
        
        # Fallback search in common locations
        search_paths = [
            filename,
            f"data/raw/{filename}",
            f"data/{filename}",
            f"examples/{filename}",
            f"../data/{filename}",
        ]
        
        print(f"🔍 Searching for {filename} in common locations...")
        for path in search_paths:
            if Path(path).exists():
                print(f"✅ Found: {path}")
                return path
            else:
                print(f"   ❌ Not found: {path}")
        
        # Show available files for debugging
        print(f"\n💡 Available CSV files in current directory:")
        csv_files = list(Path(".").glob("*.csv"))
        if csv_files:
            for csv_file in csv_files:
                print(f"   📄 {csv_file}")
        else:
            print("   (No CSV files found)")
        
        raise FileNotFoundError(
            f"Could not find {filename}. Please ensure the file exists."
        )
    
    def analyze_manual_traces(self, manual_traces_file: str) -> Dict:
        """
        Analyze existing manual traces to extract domain knowledge.
        Expected format: CSV with columns like 'Requirement_ID', 'Activity_Name', 'Requirement_Text'
        """
        print("Loading and analyzing manual traces...")
        
        # Find file using path resolution
        actual_file_path = self._find_manual_traces_file(manual_traces_file)
        
        # Load with project file handler if available
        if UTILS_AVAILABLE and self.file_handler:
            traces_df = self.file_handler.safe_read_csv(actual_file_path)
        else:
            traces_df = pd.read_csv(actual_file_path)
        
        # Basic statistics
        stats = {
            'total_traces': len(traces_df),
            'unique_requirements': traces_df['ID'].nunique() if 'ID' in traces_df.columns else len(traces_df),
            'unique_activities': traces_df['Satisfied By'].nunique() if 'Satisfied By' in traces_df.columns else 0
        }
        
        print(f"Analyzing {stats['total_traces']} manual traces...")
        
        # Extract domain knowledge
        domain_knowledge = {}
        domain_knowledge['vocabulary'] = self._extract_vocabulary_patterns(traces_df)
        domain_knowledge['phrase_patterns'] = self._extract_phrase_patterns(traces_df)
        domain_knowledge['entity_mappings'] = self._extract_entity_mappings(traces_df)
        domain_knowledge['semantic_clusters'] = self._find_semantic_clusters(traces_df)
        domain_knowledge['matching_rules'] = self._derive_matching_rules(traces_df)
        
        return {
            'statistics': stats,
            'domain_knowledge': domain_knowledge
        }
    
    def _extract_vocabulary_patterns(self, traces_df: pd.DataFrame) -> Dict:
        """Extract vocabulary mappings between requirements and activities."""
        vocab_mappings = defaultdict(set)
        synonym_pairs = []
        
        print("  Extracting vocabulary patterns...")
        
        for _, row in traces_df.iterrows():
            req_text = str(row.get('Requirement Text', ''))
            activities = str(row.get('Satisfied By', '')).split(',')
            
            if not req_text.strip():
                continue
                
            # Process requirement
            req_doc = self.nlp(req_text.lower()) if self.nlp else None
            if req_doc:
                req_tokens = {token.lemma_ for token in req_doc 
                             if not token.is_stop and token.is_alpha and len(token.text) > 2}
            else:
                # Fallback text processing
                req_tokens = set(re.findall(r'\b[a-zA-Z]{3,}\b', req_text.lower()))
            
            # Process each linked activity
            for activity in activities:
                activity = activity.strip()
                if not activity:
                    continue
                    
                if self.nlp:
                    act_doc = self.nlp(activity.lower())
                    act_tokens = {token.lemma_ for token in act_doc 
                                 if not token.is_stop and token.is_alpha and len(token.text) > 2}
                else:
                    act_tokens = set(re.findall(r'\b[a-zA-Z]{3,}\b', activity.lower()))
                
                # Find potential synonyms
                for req_token in req_tokens:
                    for act_token in act_tokens:
                        if req_token != act_token and self._are_semantically_similar(req_token, act_token):
                            synonym_pairs.append((req_token, act_token))
                            vocab_mappings[req_token].add(act_token)
        
        # Count synonym frequency
        synonym_freq = Counter(synonym_pairs)
        high_confidence_synonyms = {pair: count for pair, count in synonym_freq.items() if count >= 2}
        
        return {
            'vocabulary_mappings': {k: list(v) for k, v in vocab_mappings.items()},
            'synonym_pairs': high_confidence_synonyms,
            'total_unique_terms': len(vocab_mappings)
        }
    
    def _extract_phrase_patterns(self, traces_df: pd.DataFrame) -> Dict:
        """Extract common phrase patterns in successful matches."""
        phrase_patterns = {
            'requirement_patterns': Counter(),
            'activity_patterns': Counter(),
            'common_phrases': Counter()
        }
        
        print("  Extracting phrase patterns...")
        
        for _, row in traces_df.iterrows():
            req_text = str(row.get('Requirement Text', ''))
            activities = str(row.get('Satisfied By', '')).split(',')
            
            if not req_text.strip():
                continue
            
            # Extract requirement patterns
            if self.nlp:
                req_doc = self.nlp(req_text)
                for chunk in req_doc.noun_chunks:
                    if len(chunk.text.split()) <= 3:
                        phrase_patterns['requirement_patterns'][chunk.text.lower().strip()] += 1
            
            # Process activities
            for activity in activities:
                activity = activity.strip()
                if not activity:
                    continue
                    
                if self.nlp:
                    act_doc = self.nlp(activity)
                    for chunk in act_doc.noun_chunks:
                        if len(chunk.text.split()) <= 3:
                            phrase_patterns['activity_patterns'][chunk.text.lower().strip()] += 1
        
        # Filter to patterns that appear multiple times
        filtered_patterns = {}
        for pattern_type, patterns in phrase_patterns.items():
            filtered_patterns[pattern_type] = {phrase: count for phrase, count in patterns.items() 
                                             if count >= 2 and len(phrase.split()) > 1}
        
        return filtered_patterns
    
    def _extract_entity_mappings(self, traces_df: pd.DataFrame) -> Dict:
        """Extract named entity mappings between requirements and activities."""
        entity_mappings = defaultdict(set)
        
        print("  Extracting entity mappings...")
        
        if not self.nlp:
            return {}
        
        for _, row in traces_df.iterrows():
            req_text = str(row.get('Requirement Text', ''))
            activities = str(row.get('Satisfied By', '')).split(',')
            
            if not req_text.strip():
                continue
            
            req_doc = self.nlp(req_text)
            req_entities = {(ent.text.lower(), ent.label_) for ent in req_doc.ents}
            
            for activity in activities:
                activity = activity.strip()
                if not activity:
                    continue
                    
                act_doc = self.nlp(activity)
                act_entities = {(ent.text.lower(), ent.label_) for ent in act_doc.ents}
                
                # Map entities of same type
                for req_ent_text, req_ent_type in req_entities:
                    for act_ent_text, act_ent_type in act_entities:
                        if req_ent_type == act_ent_type and req_ent_text != act_ent_text:
                            entity_mappings[req_ent_text].add(act_ent_text)
        
        return {entity: list(mappings) for entity, mappings in entity_mappings.items() if len(mappings) > 0}
    
    def _find_semantic_clusters(self, traces_df: pd.DataFrame) -> Dict:
        """Find semantic clusters of related requirements and activities."""
        print("  Finding semantic clusters...")
        
        # Collect all requirement texts
        req_texts = []
        
        for _, row in traces_df.iterrows():
            req_text = str(row.get('Requirement Text', ''))
            if req_text.strip():
                req_texts.append(req_text)
        
        # Simple clustering without sklearn dependency
        clusters = []
        if len(req_texts) > 1:
            # Basic similarity clustering
            for i, text1 in enumerate(req_texts[:10]):  # Limit for performance
                similar_texts = []
                for j, text2 in enumerate(req_texts):
                    if i != j and self._simple_text_similarity(text1, text2) > 0.3:
                        similar_texts.append(text2)
                
                if similar_texts:
                    clusters.append({
                        'anchor': text1,
                        'similar_requirements': similar_texts[:3]
                    })
        
        return {'requirement_clusters': clusters[:10]}
    
    def _derive_matching_rules(self, traces_df: pd.DataFrame) -> Dict:
        """Derive high-level matching rules from successful traces."""
        print("  Deriving matching rules...")
        
        rules = {
            'strong_indicators': Counter(),
            'weak_indicators': Counter(),
            'anti_patterns': Counter()
        }
        
        # Simple rule extraction
        for _, row in traces_df.iterrows():
            req_text = str(row.get('Requirement Text', ''))
            activities = str(row.get('Satisfied By', '')).split(',')
            
            if not req_text.strip():
                continue
            
            req_words = set(re.findall(r'\b[a-zA-Z]{3,}\b', req_text.lower()))
            
            for activity in activities:
                activity = activity.strip()
                if not activity:
                    continue
                    
                act_words = set(re.findall(r'\b[a-zA-Z]{3,}\b', activity.lower()))
                
                # Shared words are strong indicators
                shared_words = req_words & act_words
                for word in shared_words:
                    rules['strong_indicators'][f"shared_word:{word}"] += 1
        
        # Filter rules by frequency
        filtered_rules = {}
        for rule_type, rule_counts in rules.items():
            filtered_rules[rule_type] = {rule: count for rule, count in rule_counts.items() if count >= 3}
        
        return filtered_rules
    
    def _are_semantically_similar(self, word1: str, word2: str, threshold: float = 0.3) -> bool:
        """Check if two words are semantically similar using spaCy vectors."""
        if not self.nlp:
            # Fallback: simple character similarity
            if abs(len(word1) - len(word2)) > 3:
                return False
            common_chars = set(word1) & set(word2)
            return len(common_chars) >= 3
        
        try:
            token1 = self.nlp(word1)[0]
            token2 = self.nlp(word2)[0]
            
            if token1.has_vector and token2.has_vector:
                similarity = token1.similarity(token2)
                return similarity > threshold
        except:
            pass
        return False
    
    def _simple_text_similarity(self, text1: str, text2: str) -> float:
        """Simple text similarity without external dependencies."""
        words1 = set(re.findall(r'\b[a-zA-Z]{3,}\b', text1.lower()))
        words2 = set(re.findall(r'\b[a-zA-Z]{3,}\b', text2.lower()))
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0
    
    def save_domain_knowledge(self, domain_knowledge: Dict, output_file: str = "domain_knowledge.json"):
        """Save extracted domain knowledge to file."""
        # Convert Counter objects to regular dicts for JSON serialization
        def convert_counters(obj):
            if isinstance(obj, Counter):
                return dict(obj)
            elif isinstance(obj, dict):
                return {k: convert_counters(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_counters(item) for item in obj]
            else:
                return obj
        
        serializable_knowledge = convert_counters(domain_knowledge)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_knowledge, f, indent=2, ensure_ascii=False)
        
        print(f"Domain knowledge saved to {output_file}")
    
    def create_enhanced_synonyms(self, domain_knowledge: Dict, 
                               existing_synonyms_file: str = "synonyms.json") -> Dict:
        """Create enhanced synonym dictionary from domain knowledge."""
        # Load existing synonyms
        try:
            with open(existing_synonyms_file, 'r') as f:
                synonyms = json.load(f)
        except FileNotFoundError:
            synonyms = {}
        
        # Add domain-specific vocabulary mappings
        vocab_mappings = domain_knowledge.get('domain_knowledge', {}).get('vocabulary', {}).get('vocabulary_mappings', {})
        
        for term, similar_terms in vocab_mappings.items():
            if term not in synonyms:
                synonyms[term] = []
            
            for similar_term in similar_terms:
                if similar_term not in synonyms[term]:
                    synonyms[term].append(similar_term)
        
        return synonyms


# Main function
def main():
    """Example usage of domain knowledge builder."""
    print("Building Domain Knowledge from Manual Traces")
    print("=" * 50)
    
    # Initialize builder
    builder = DomainKnowledgeBuilder()
    
    try:
        # Analyze existing manual traces
        domain_analysis = builder.analyze_manual_traces("manual_matches.csv")
        
        # Print summary
        stats = domain_analysis['statistics']
        knowledge = domain_analysis['domain_knowledge']
        
        print(f"\nAnalysis Summary:")
        print(f"  Total traces analyzed: {stats['total_traces']}")
        print(f"  Unique requirements: {stats['unique_requirements']}")
        print(f"  Unique activities: {stats['unique_activities']}")
        
        # Save domain knowledge
        builder.save_domain_knowledge(domain_analysis, "extracted_domain_knowledge.json")
        
        # Create enhanced synonyms
        enhanced_synonyms = builder.create_enhanced_synonyms(domain_analysis)
        with open("enhanced_synonyms.json", 'w') as f:
            json.dump(enhanced_synonyms, f, indent=2)
        
        print(f"\nEnhanced synonym dictionary created with {len(enhanced_synonyms)} terms")
        
    except FileNotFoundError as e:
        print(f"\n❌ File not found: {e}")
        print(f"💡 The file should be in one of these locations:")
        print(f"   - Current directory: ./manual_matches.csv")
        print(f"   - Data directory: ./data/raw/manual_matches.csv")
        print(f"   - Or specify full path: python domain_knowledge_builder.py")


if __name__ == "__main__":
    main()