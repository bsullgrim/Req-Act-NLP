"""
Domain Resources - Manage aerospace vocabulary and term expansion
Loads external JSON files with fallback to baseline resources
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Set, Optional

logger = logging.getLogger(__name__)

class DomainResources:
    """Manages domain vocabulary and term expansion from external resources."""
    
    def __init__(self, domain_path: str = "resources/aerospace"):
        """Initialize with domain resource files."""
        self.domain_path = Path(domain_path)
        self.domain_knowledge_path = self.domain_path / "domain_knowledge"
        
        # Initialize empty containers
        self.vocabulary = {}
        self.synonyms = {}
        self.abbreviations = {}
        
        # NEW: Initialize containers for additional domain knowledge
        self.phrase_patterns = {
            'requirement_patterns': {},
            'activity_patterns': {}
        }
        self.matching_rules = {
            'strong_indicators': {}
        }
        self.cooccurrence_terms = {}
        
        # Load all resources with fallback logic
        self._load_resources()
        
        # NEW: Load extracted domain knowledge if available
        self._load_extracted_domain_knowledge()
        
        # Verify critical functionality
        self._verify_functionality()
        
        logger.info(f"✅ Domain resources loaded from {domain_path}")
    
    def _load_resources(self):
        """Load all JSON resource files with fallback logic."""
        try:
            # 1. Load vocabulary (baseline only for now)
            self.vocabulary = self._load_json("vocabulary.json", use_enhanced=False)
            
            # 2. Load synonyms with fallback
            self.synonyms = self._load_json_with_fallback("synonyms.json", "learned_synonyms.json")
            
            # 3. Load abbreviations (baseline only for now)
            self.abbreviations = self._load_json("abbreviations.json", use_enhanced=False)
            
            # Log loading success
            logger.info(f"📚 Loaded {len(self.vocabulary)} vocabulary categories")
            logger.info(f"🔗 Loaded {len(self.synonyms)} synonym entries")
            logger.info(f"📝 Loaded {len(self.abbreviations)} abbreviations")
            
            # Log which resources are enhanced vs baseline
            if (self.domain_knowledge_path / "learned_synonyms.json").exists():
                enhanced_path = self.domain_knowledge_path / "learned_synonyms.json"
                try:
                    with open(enhanced_path, 'r', encoding='utf-8') as f:
                        enhanced_data = json.load(f)
                        if enhanced_data:  # Not empty
                            logger.info("✨ Using ENHANCED synonyms from domain knowledge")
                        else:
                            logger.info("📋 Using baseline synonyms (enhanced file empty)")
                except:
                    logger.info("📋 Using baseline synonyms")
            else:
                logger.info("📋 Using baseline synonyms (no enhanced version)")
            
        except Exception as e:
            logger.error(f"❌ Failed to load domain resources: {e}")
            # Fallback to minimal hardcoded resources
            self._create_fallback_resources()
    
    def _load_json_with_fallback(self, baseline_filename: str, enhanced_filename: str) -> Dict:
        """Load JSON with fallback: try enhanced first, then baseline."""
        # Try enhanced version first
        enhanced_path = self.domain_knowledge_path / enhanced_filename
        if enhanced_path.exists():
            try:
                with open(enhanced_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if data:  # Not empty
                        logger.debug(f"✅ Loaded enhanced {enhanced_filename}: {len(data)} entries")
                        return data
            except Exception as e:
                logger.warning(f"⚠️ Could not load enhanced {enhanced_filename}: {e}")
        
        # Fallback to baseline
        return self._load_json(baseline_filename, use_enhanced=False)
    
    def _load_json(self, filename: str, use_enhanced: bool = True) -> Dict:
        """Load a JSON file from the domain path."""
        # Determine path
        if use_enhanced and (self.domain_knowledge_path / filename).exists():
            file_path = self.domain_knowledge_path / filename
        else:
            file_path = self.domain_path / filename
        
        if not file_path.exists():
            logger.warning(f"⚠️ Resource file not found: {file_path}")
            return {}
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                logger.debug(f"✅ Loaded {filename}: {len(data)} entries from {file_path.parent.name}")
                return data
        except Exception as e:
            logger.error(f"❌ Failed to load {filename}: {e}")
            return {}
    
    def _create_fallback_resources(self):
        """Create minimal fallback resources if files can't be loaded."""
        logger.warning("🔄 Using fallback resources...")
        
        self.synonyms = {
            "monitor": ["track", "observe", "watch"],
            "control": ["manage", "regulate", "command"],
            "transmit": ["send", "broadcast", "relay"],
            "receive": ["acquire", "capture", "obtain"],
            "system": ["subsystem", "component", "module"]
        }
        
        self.abbreviations = {
            "acs": "attitude control system",
            "comm": "communication",
            "nav": "navigation",
            "cmd": "command"
        }
        
        self.vocabulary = {
            "systems": ["system", "subsystem", "component"],
            "operations": ["monitor", "control", "transmit", "receive"]
        }

    def _verify_functionality(self):
        """Verify that expand_terms will work correctly."""
        test_terms = ["monitor", "control", "system"]
        expanded = self.expand_terms(test_terms)
        
        if len(expanded) <= len(test_terms):
            logger.warning(f"⚠️ expand_terms() not expanding: {test_terms} → {expanded}")
        else:
            logger.info(f"✅ expand_terms() working: {len(test_terms)} → {len(expanded)} terms")
    
    def _load_extracted_domain_knowledge(self):
        """Load additional domain knowledge from extracted_domain_knowledge.json."""
        try:
            # Try multiple locations for the extracted domain knowledge file
            possible_paths = [
                Path("extracted_domain_knowledge.json"),
                self.domain_knowledge_path / "extracted_domain_knowledge.json",
                self.domain_path / "extracted_domain_knowledge.json"
            ]
            
            domain_knowledge_data = None
            for path in possible_paths:
                if path.exists():
                    with open(path, 'r', encoding='utf-8') as f:
                        domain_knowledge_data = json.load(f)
                        logger.info(f"📚 Loaded extracted domain knowledge from {path}")
                        break
            
            if domain_knowledge_data:
                # Extract phrase patterns
                if 'domain_knowledge' in domain_knowledge_data:
                    dk = domain_knowledge_data['domain_knowledge']
                    
                    # Load phrase patterns
                    if 'phrase_patterns' in dk:
                        self.phrase_patterns = dk['phrase_patterns']
                        total_phrases = len(self.phrase_patterns.get('requirement_patterns', {})) + \
                                      len(self.phrase_patterns.get('activity_patterns', {}))
                        logger.info(f"📝 Loaded {total_phrases} phrase patterns")
                    
                    # Load matching rules
                    if 'matching_rules' in dk:
                        self.matching_rules = dk['matching_rules']
                        strong_indicators = self.matching_rules.get('strong_indicators', {})
                        logger.info(f"🎯 Loaded {len(strong_indicators)} matching rules")
                    
                    # Load cooccurrence terms (for richer domain scoring)
                    if 'vocabulary' in dk and 'cooccurrence_synonyms' in dk['vocabulary']:
                        self.cooccurrence_terms = dk['vocabulary']['cooccurrence_synonyms']
                        logger.info(f"🔗 Loaded {len(self.cooccurrence_terms)} cooccurrence term mappings")
            else:
                logger.info("📋 No extracted domain knowledge found - using baseline resources only")
                
        except Exception as e:
            logger.warning(f"⚠️ Could not load extracted domain knowledge: {e}")
    
    def get_phrase_patterns(self) -> Dict[str, Dict[str, int]]:
        """Get phrase patterns from domain knowledge."""
        return self.phrase_patterns
    
    def get_matching_rules(self) -> Dict[str, Dict[str, int]]:
        """Get matching rules from domain knowledge."""
        return self.matching_rules
    
    def get_strong_indicator_words(self) -> Set[str]:
        """Get words that strongly indicate requirement-activity matches."""
        strong_indicators = self.matching_rules.get('strong_indicators', {})
        # Extract just the words (remove 'shared_word:' prefix)
        words = set()
        for indicator in strong_indicators:
            if indicator.startswith('shared_word:'):
                word = indicator.replace('shared_word:', '')
                words.add(word)
        return words
    
    def check_phrase_overlap(self, text1: str, text2: str) -> float:
        """Check if important phrases appear in both texts."""
        text1_lower = text1.lower()
        text2_lower = text2.lower()
        
        score = 0.0
        
        # Check requirement patterns
        for phrase, count in self.phrase_patterns.get('requirement_patterns', {}).items():
            if count >= 2:  # Only consider phrases that appear at least twice
                if phrase in text1_lower and phrase in text2_lower:
                    # Higher score for more frequent phrases
                    score += 0.1 * (1 + min(count / 10, 1))
        
        # Check activity patterns
        for phrase, count in self.phrase_patterns.get('activity_patterns', {}).items():
            if count >= 2:
                if phrase in text1_lower and phrase in text2_lower:
                    score += 0.1 * (1 + min(count / 10, 1))
        
        return min(score, 1.0)
    
    def get_domain_cooccurrence_score(self, req_terms: Set[str], act_terms: Set[str]) -> float:
        """Score based on learned term co-occurrences."""
        score = 0.0
        
        # Check if requirement terms have activity terms as cooccurrence partners
        for req_term in req_terms:
            if req_term in self.cooccurrence_terms:
                cooccurring = set(self.cooccurrence_terms[req_term])
                overlap = cooccurring & act_terms
                if overlap:
                    # More overlap = higher score
                    score += 0.1 * len(overlap)
        
        return min(score, 1.0)
    
    def get_resource_status(self) -> Dict[str, str]:
        """Get status of which resources are being used."""
        status = {}
        
        # Check synonyms
        enhanced_synonyms_path = self.domain_knowledge_path / "learned_synonyms.json"
        if enhanced_synonyms_path.exists():
            try:
                with open(enhanced_synonyms_path, 'r') as f:
                    data = json.load(f)
                    if data:
                        status['synonyms'] = f"Enhanced ({len(data)} terms)"
                    else:
                        status['synonyms'] = "Baseline (enhanced file empty)"
            except:
                status['synonyms'] = "Baseline (enhanced file error)"
        else:
            status['synonyms'] = "Baseline (no enhanced file)"
        
        status['vocabulary'] = f"Baseline ({len(self.vocabulary)} categories)"
        status['abbreviations'] = f"Baseline ({len(self.abbreviations)} terms)"
        
        return status
    
    # [Keep all other methods unchanged - expand_terms, get_synonyms, etc.]
    
    def get_domain_terms(self, category: Optional[str] = None) -> Set[str]:
        """Get domain vocabulary terms."""
        if category and category in self.vocabulary:
            return set(self.vocabulary[category])
        
        # Return all terms from all categories
        all_terms = set()
        for category_terms in self.vocabulary.values():
            if isinstance(category_terms, list):
                all_terms.update(category_terms)
        return all_terms
    
    def expand_terms(self, terms: List[str]) -> List[str]:
        """
        CRITICAL FIX: Expand terms with synonyms and abbreviations.
        This method is called by matcher.py expand_query_aerospace()
        
        Args:
            terms: List of terms to expand
            
        Returns:
            Original terms plus all expansions (unique)
        """
        if not terms:
            return []
        
        expanded = set()
        
        # Add original terms
        for term in terms:
            if term and term.strip():
                expanded.add(term.lower().strip())
        
        # Expand each term
        for term in terms:
            if not term or not term.strip():
                continue
                
            term_lower = term.lower().strip()
            
            # Expand abbreviations
            if term_lower in self.abbreviations:
                full_form = self.abbreviations[term_lower]
                if full_form and full_form.strip():
                    expanded.add(full_form.strip())
                    logger.debug(f"🔤 Expanded abbrev '{term}' → '{full_form}'")
            
            # Expand synonyms
            if term_lower in self.synonyms:
                synonyms = self.synonyms[term_lower]
                if synonyms and isinstance(synonyms, list):
                    for synonym in synonyms:
                        if synonym and synonym.strip():
                            expanded.add(synonym.strip())
                    logger.debug(f"🔗 Added {len(synonyms)} synonyms for '{term}'")
        
        result = list(expanded)
        
        # Log expansion results
        if len(result) > len(terms):
            logger.debug(f"📈 Successfully expanded {len(terms)} → {len(result)} terms")
        else:
            logger.warning(f"⚠️ No expansion: {terms} → {result}")
        
        return result
        
    def get_abbreviation(self, abbrev: str) -> Optional[str]:
        """Get full form of abbreviation."""
        if not abbrev:
            return None
        return self.abbreviations.get(abbrev.lower().strip())
    
    def get_synonyms(self, term: str) -> List[str]:
        """Get synonyms for a term."""
        if not term:
            return []
        synonyms = self.synonyms.get(term.lower().strip(), [])
        return synonyms if isinstance(synonyms, list) else []
    
    def is_domain_term(self, term: str) -> bool:
        """Check if a term is in the domain vocabulary."""
        if not term:
            return False
        term_lower = term.lower().strip()
        return term_lower in self.get_domain_terms()
    
    def get_vocabulary_stats(self) -> Dict[str, int]:
        """Get statistics about loaded vocabulary."""
        stats = {
            'total_categories': len(self.vocabulary),
            'total_terms': len(self.get_domain_terms()),
            'total_synonyms': len(self.synonyms),
            'total_abbreviations': len(self.abbreviations)
        }
        
        # Add per-category counts
        for category, terms in self.vocabulary.items():
            if isinstance(terms, list):
                stats[f'{category}_terms'] = len(terms)
        
        return stats
    
    def reload_resources(self):
        """Reload all resource files (useful for development/testing)."""
        logger.info("🔄 Reloading domain resources...")
        self._load_resources()

    def test_expand_terms(self, test_terms: List[str] = None) -> Dict:
        """Test expand_terms functionality and return diagnostics."""
        if test_terms is None:
            test_terms = ["monitor", "control", "acs", "transmit", "system"]
        
        print(f"🧪 Testing expand_terms() with: {test_terms}")
        
        expanded = self.expand_terms(test_terms)
        
        diagnostics = {
            'input_terms': test_terms,
            'input_count': len(test_terms),
            'output_terms': expanded,
            'output_count': len(expanded),
            'expansion_ratio': len(expanded) / len(test_terms) if test_terms else 0,
            'working': len(expanded) > len(test_terms),
            'added_terms': [term for term in expanded if term not in [t.lower() for t in test_terms]]
        }
        
        print(f"📊 Results:")
        print(f"   Input: {test_terms} ({len(test_terms)} terms)")
        print(f"   Output: {expanded} ({len(expanded)} terms)")
        print(f"   Expansion ratio: {diagnostics['expansion_ratio']:.1f}x")
        print(f"   Working: {'✅ YES' if diagnostics['working'] else '❌ NO'}")
        print(f"   Added terms: {diagnostics['added_terms']}")
        
        # Show resource status
        print(f"\n📁 Resource Status:")
        status = self.get_resource_status()
        for resource, info in status.items():
            print(f"   {resource}: {info}")
        
        return diagnostics

def test_domain_resources():
    """Test the domain resources thoroughly."""
    print("🚀 COMPREHENSIVE DOMAIN RESOURCES TEST")
    print("=" * 60)
    
    # Test with default path
    try:
        resources = DomainResources()
        stats = resources.get_vocabulary_stats()
        print(f"📊 Vocabulary stats: {stats}")
        
        # Test expand_terms (THE CRITICAL METHOD)
        test_result = resources.test_expand_terms()
        
        return test_result['working']
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    working = test_domain_resources()
    if working:
        print(f"\n🎉 SUCCESS: Domain resources working correctly!")
        print(f"🎯 expand_terms() will fix query expansion in matcher.py")
    else:
        print(f"\n❌ FAILURE: Domain resources need fixing")
        print(f"💡 Check resource files in resources/aerospace/")