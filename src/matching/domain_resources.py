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
        
        # Load all resources with fallback logic
        self._load_resources()
        
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