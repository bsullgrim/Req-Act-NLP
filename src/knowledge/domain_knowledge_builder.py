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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

class DomainKnowledgeBuilder:
    """Extract domain knowledge from existing manual requirement-activity traces."""
    
    def __init__(self, spacy_model: str = "en_core_web_sm"):
        self.nlp = spacy.load(spacy_model)
        self.domain_patterns = {}
        self.vocabulary_mappings = {}
        self.phrase_patterns = {}
        self.entity_mappings = {}
        
    def analyze_manual_traces(self, manual_traces_file: str) -> Dict:
        """
        Analyze existing manual traces to extract domain knowledge.
        Expected format: CSV with columns like 'Requirement_ID', 'Activity_Name', 'Requirement_Text'
        """
        print("Loading and analyzing manual traces...")
        
        # Load manual traces
        traces_df = pd.read_csv(manual_traces_file)
        
        # Basic statistics
        stats = {
            'total_traces': len(traces_df),
            'unique_requirements': traces_df['ID'].nunique() if 'ID' in traces_df.columns else len(traces_df),
            'unique_activities': traces_df['Satisfied By'].nunique() if 'Satisfied By' in traces_df.columns else 0
        }
        
        print(f"Analyzing {stats['total_traces']} manual traces...")
        
        # Extract domain knowledge
        domain_knowledge = {}
        
        # 1. Vocabulary Analysis
        domain_knowledge['vocabulary'] = self._extract_vocabulary_patterns(traces_df)
        
        # 2. Phrase Patterns
        domain_knowledge['phrase_patterns'] = self._extract_phrase_patterns(traces_df)
        
        # 3. Entity Mappings
        domain_knowledge['entity_mappings'] = self._extract_entity_mappings(traces_df)
        
        # 4. Semantic Clusters
        domain_knowledge['semantic_clusters'] = self._find_semantic_clusters(traces_df)
        
        # 5. Matching Rules
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
            req_doc = self.nlp(req_text.lower())
            req_tokens = {token.lemma_ for token in req_doc 
                         if not token.is_stop and token.is_alpha and len(token.text) > 2}
            
            # Process each linked activity
            for activity in activities:
                activity = activity.strip()
                if not activity:
                    continue
                    
                act_doc = self.nlp(activity.lower())
                act_tokens = {token.lemma_ for token in act_doc 
                             if not token.is_stop and token.is_alpha and len(token.text) > 2}
                
                # Find vocabulary overlaps
                common_tokens = req_tokens & act_tokens
                
                # Find potential synonyms (tokens that co-occur but are different)
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
            req_doc = self.nlp(req_text)
            
            # Noun phrases
            for chunk in req_doc.noun_chunks:
                if len(chunk.text.split()) <= 3:  # Reasonable phrase length
                    phrase_patterns['requirement_patterns'][chunk.text.lower().strip()] += 1
            
            # Verb phrases (simple: verb + object)
            for token in req_doc:
                if token.pos_ == "VERB" and token.dep_ == "ROOT":
                    objects = [child.text for child in token.children if child.dep_ in ["dobj", "pobj"]]
                    if objects:
                        verb_phrase = f"{token.lemma_} {' '.join(objects)}"
                        phrase_patterns['requirement_patterns'][verb_phrase.lower()] += 1
            
            # Process activities
            for activity in activities:
                activity = activity.strip()
                if not activity:
                    continue
                    
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
        act_texts = []
        
        for _, row in traces_df.iterrows():
            req_text = str(row.get('Requirement Text', ''))
            activities = str(row.get('Satisfied By', '')).split(',')
            
            if req_text.strip():
                req_texts.append(req_text)
                
            for activity in activities:
                activity = activity.strip()
                if activity:
                    act_texts.append(activity)
        
        # Use TF-IDF to find clusters
        if len(req_texts) > 1:
            vectorizer = TfidfVectorizer(max_features=1000, stop_words='english', ngram_range=(1, 2))
            try:
                req_vectors = vectorizer.fit_transform(req_texts)
                
                # Simple clustering: find highly similar pairs
                similarity_matrix = cosine_similarity(req_vectors)
                
                clusters = []
                threshold = 0.5  # Similarity threshold
                
                for i in range(len(req_texts)):
                    similar_indices = np.where(similarity_matrix[i] > threshold)[0]
                    if len(similar_indices) > 1:  # More than just itself
                        cluster = [req_texts[j] for j in similar_indices if j != i]
                        if cluster:
                            clusters.append({
                                'anchor': req_texts[i],
                                'similar_requirements': cluster[:3],  # Limit to top 3
                                'similarity_scores': [similarity_matrix[i][j] for j in similar_indices if j != i][:3]
                            })
                
                return {'requirement_clusters': clusters[:10]}  # Top 10 clusters
            except:
                return {'requirement_clusters': []}
        
        return {'requirement_clusters': []}
    
    def _derive_matching_rules(self, traces_df: pd.DataFrame) -> Dict:
        """Derive high-level matching rules from successful traces."""
        print("  Deriving matching rules...")
        
        rules = {
            'strong_indicators': Counter(),
            'weak_indicators': Counter(),
            'anti_patterns': Counter()
        }
        
        # Analyze successful matches for patterns
        for _, row in traces_df.iterrows():
            req_text = str(row.get('Requirement Text', ''))
            activities = str(row.get('Satisfied By', '')).split(',')
            
            if not req_text.strip():
                continue
            
            req_doc = self.nlp(req_text.lower())
            
            for activity in activities:
                activity = activity.strip()
                if not activity:
                    continue
                    
                act_doc = self.nlp(activity.lower())
                
                # Rule: Direct verb matches are strong indicators
                req_verbs = {token.lemma_ for token in req_doc if token.pos_ == "VERB"}
                act_verbs = {token.lemma_ for token in act_doc if token.pos_ == "VERB"}
                verb_overlap = req_verbs & act_verbs
                
                for verb in verb_overlap:
                    rules['strong_indicators'][f"shared_verb:{verb}"] += 1
                
                # Rule: Shared technical nouns
                req_nouns = {token.lemma_ for token in req_doc 
                           if token.pos_ in ["NOUN", "PROPN"] and len(token.text) > 3}
                act_nouns = {token.lemma_ for token in act_doc 
                           if token.pos_ in ["NOUN", "PROPN"] and len(token.text) > 3}
                noun_overlap = req_nouns & act_nouns
                
                for noun in noun_overlap:
                    rules['strong_indicators'][f"shared_noun:{noun}"] += 1
        
        # Filter rules by frequency
        filtered_rules = {}
        for rule_type, rule_counts in rules.items():
            filtered_rules[rule_type] = {rule: count for rule, count in rule_counts.items() if count >= 3}
        
        return filtered_rules
    
    def _are_semantically_similar(self, word1: str, word2: str, threshold: float = 0.6) -> bool:
        """Check if two words are semantically similar using spaCy vectors."""
        try:
            token1 = self.nlp(word1)[0]
            token2 = self.nlp(word2)[0]
            
            if token1.has_vector and token2.has_vector:
                similarity = token1.similarity(token2)
                return similarity > threshold
        except:
            pass
        return False
    
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
        
        # Add high-confidence synonym pairs
        synonym_pairs = domain_knowledge.get('domain_knowledge', {}).get('vocabulary', {}).get('synonym_pairs', {})
        
        for (term1, term2), frequency in synonym_pairs.items():
            if frequency >= 3:  # High confidence threshold
                if term1 not in synonyms:
                    synonyms[term1] = []
                if term2 not in synonyms[term1]:
                    synonyms[term1].append(term2)
                
                if term2 not in synonyms:
                    synonyms[term2] = []
                if term1 not in synonyms[term2]:
                    synonyms[term2].append(term1)
        
        return synonyms


def main():
    """Example usage of domain knowledge builder."""
    print("Building Domain Knowledge from Manual Traces")
    print("=" * 50)
    
    # Initialize builder
    builder = DomainKnowledgeBuilder()
    
    # Analyze existing manual traces
    domain_analysis = builder.analyze_manual_traces("manual_matches.csv")
    
    # Print summary
    stats = domain_analysis['statistics']
    knowledge = domain_analysis['domain_knowledge']
    
    print(f"\nAnalysis Summary:")
    print(f"  Total traces analyzed: {stats['total_traces']}")
    print(f"  Unique requirements: {stats['unique_requirements']}")
    print(f"  Unique activities: {stats['unique_activities']}")
    
    print(f"\nDomain Knowledge Extracted:")
    
    vocab_info = knowledge.get('vocabulary', {})
    print(f"  Vocabulary mappings: {vocab_info.get('total_unique_terms', 0)}")
    print(f"  Synonym pairs found: {len(vocab_info.get('synonym_pairs', {}))}")
    
    phrase_info = knowledge.get('phrase_patterns', {})
    req_patterns = len(phrase_info.get('requirement_patterns', {}))
    act_patterns = len(phrase_info.get('activity_patterns', {}))
    print(f"  Requirement patterns: {req_patterns}")
    print(f"  Activity patterns: {act_patterns}")
    
    entity_info = knowledge.get('entity_mappings', {})
    print(f"  Entity mappings: {len(entity_info)}")
    
    clusters = knowledge.get('semantic_clusters', {}).get('requirement_clusters', [])
    print(f"  Semantic clusters: {len(clusters)}")
    
    rules = knowledge.get('matching_rules', {})
    strong_rules = len(rules.get('strong_indicators', {}))
    print(f"  Strong matching rules: {strong_rules}")
    
    # Save domain knowledge
    builder.save_domain_knowledge(domain_analysis, "extracted_domain_knowledge.json")
    
    # Create enhanced synonyms
    enhanced_synonyms = builder.create_enhanced_synonyms(domain_analysis)
    with open("enhanced_synonyms.json", 'w') as f:
        json.dump(enhanced_synonyms, f, indent=2)
    
    print(f"\nEnhanced synonym dictionary created with {len(enhanced_synonyms)} terms")
    
    print(f"\nNext steps:")
    print(f"1. Review extracted_domain_knowledge.json")
    print(f"2. Use enhanced_synonyms.json in your matcher")
    print(f"3. Consider the identified patterns for custom scoring")


if __name__ == "__main__":
    main()