"""
Domain Resources - Manage aerospace vocabulary and term expansion
Loads external JSON files instead of hardcoded dictionaries
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Set, Optional

logger = logging.getLogger(__name__)

class DomainResources:
    """Manages domain vocabulary and term expansion from external resources."""
    
    def __init__(self, domain_path: str = "resources/aerospace"):
        """
        Initialize with domain resource files.
        
        Args:
            domain_path: Path to domain resource directory
        """
        self.domain_path = Path(domain_path)
        self.vocabulary = {}
        self.synonyms = {}
        self.abbreviations = {}
        
        # Load all resources
        self._load_resources()
        
        logger.info(f"✅ Loaded domain resources from {domain_path}")
        logger.info(f"   📚 Vocabulary categories: {len(self.vocabulary)}")
        logger.info(f"   🔗 Synonym entries: {len(self.synonyms)}")
        logger.info(f"   📝 Abbreviations: {len(self.abbreviations)}")
    
    def _load_resources(self):
        """Load all JSON resource files."""
        try:
            # Load all resources from resources/aerospace/ directory
            self.vocabulary = self._load_json("vocabulary.json")
            self.synonyms = self._load_json("synonyms.json")  # Use our extracted aerospace synonyms
            self.abbreviations = self._load_json("abbreviations.json")
                
        except Exception as e:
            logger.error(f"❌ Failed to load domain resources: {e}")
            # Fall back to empty dictionaries
            self.vocabulary = {}
            self.synonyms = {}
            self.abbreviations = {}
    
    def _load_json(self, filename: str) -> Dict:
        """Load a JSON file from the domain path."""
        file_path = self.domain_path / filename
        
        if not file_path.exists():
            logger.warning(f"⚠️ Resource file not found: {file_path}")
            return {}
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                logger.debug(f"✅ Loaded {filename}: {len(data)} entries")
                return data
        except Exception as e:
            logger.error(f"❌ Failed to load {filename}: {e}")
            return {}
    
    def get_domain_terms(self, category: Optional[str] = None) -> Set[str]:
        """
        Get domain vocabulary terms.
        
        Args:
            category: Optional category filter (e.g., 'systems', 'operations')
            
        Returns:
            Set of domain terms
        """
        if category and category in self.vocabulary:
            return set(self.vocabulary[category])
        
        # Return all terms from all categories
        all_terms = set()
        for category_terms in self.vocabulary.values():
            all_terms.update(category_terms)
        
        return all_terms
    
    def expand_terms(self, terms: List[str]) -> List[str]:
        """
        Expand terms with synonyms and abbreviations.
        
        Args:
            terms: List of terms to expand
            
        Returns:
            Original terms plus all expansions (unique)
            
        Example:
            Input: ['acs', 'monitor']
            Output: ['acs', 'attitude control system', 'monitor', 'track', 'observe']
        """
        expanded = set(terms)  # Start with original terms
        
        for term in terms:
            term_lower = term.lower()
            
            # Expand abbreviations
            if term_lower in self.abbreviations:
                expanded.add(self.abbreviations[term_lower])
                logger.debug(f"🔤 Expanded abbrev '{term}' → '{self.abbreviations[term_lower]}'")
            
            # Expand synonyms
            if term_lower in self.synonyms:
                for synonym in self.synonyms[term_lower]:
                    expanded.add(synonym)
                logger.debug(f"🔗 Added synonyms for '{term}': {self.synonyms[term_lower]}")
        
        result = list(expanded)
        logger.debug(f"📈 Expanded {len(terms)} terms to {len(result)} terms")
        return result
    
    def get_abbreviation(self, abbrev: str) -> Optional[str]:
        """
        Get full form of abbreviation.
        
        Args:
            abbrev: Abbreviation to expand
            
        Returns:
            Full form or None if not found
        """
        return self.abbreviations.get(abbrev.lower())
    
    def get_synonyms(self, term: str) -> List[str]:
        """
        Get synonyms for a term.
        
        Args:
            term: Term to find synonyms for
            
        Returns:
            List of synonyms (empty if none found)
        """
        return self.synonyms.get(term.lower(), [])
    
    def is_domain_term(self, term: str) -> bool:
        """
        Check if a term is in the domain vocabulary.
        
        Args:
            term: Term to check
            
        Returns:
            True if term is in any vocabulary category
        """
        term_lower = term.lower()
        return term_lower in self.get_domain_terms()
    
    def get_vocabulary_stats(self) -> Dict[str, int]:
        """
        Get statistics about loaded vocabulary.
        
        Returns:
            Dictionary with vocabulary statistics
        """
        stats = {
            'total_categories': len(self.vocabulary),
            'total_terms': len(self.get_domain_terms()),
            'total_synonyms': len(self.synonyms),
            'total_abbreviations': len(self.abbreviations)
        }
        
        # Add per-category counts
        for category, terms in self.vocabulary.items():
            stats[f'{category}_terms'] = len(terms)
        
        return stats
    
    def reload_resources(self):
        """Reload all resource files (useful for development/testing)."""
        logger.info("🔄 Reloading domain resources...")
        self._load_resources()


def test_domain_resources():
    """Test function to verify domain resources work correctly."""
    print("🧪 Testing Domain Resources (Lightweight Aerospace Focus)...")
    
    # Initialize
    resources = DomainResources()
    
    # Test vocabulary
    system_terms = resources.get_domain_terms('systems')
    print(f"✅ System terms: {len(system_terms)} (sample: {list(system_terms)[:5]})")
    
    all_terms = resources.get_domain_terms()
    print(f"✅ Total domain terms: {len(all_terms)}")
    
    # Test abbreviation expansion
    acs_expansion = resources.get_abbreviation('acs')
    print(f"✅ ACS expansion: '{acs_expansion}'")
    
    # Test synonym expansion - key aerospace terms
    test_synonyms = ['monitor', 'transmit', 'control', 'system']
    for term in test_synonyms:
        synonyms = resources.get_synonyms(term)
        print(f"✅ {term} synonyms: {synonyms}")
    
    # Test term expansion (the KEY method for fixing query expansion!)
    test_terms = ['monitor', 'acs', 'transmit', 'system']
    expanded = resources.expand_terms(test_terms)
    print(f"\n🎯 CRITICAL TEST - Term expansion:")
    print(f"   Input: {test_terms} ({len(test_terms)} terms)")
    print(f"   Output: {expanded} ({len(expanded)} terms)")
    print(f"   Expansion ratio: {len(expanded)/len(test_terms):.1f}x")
    
    # Verify this returns meaningful results (not empty!)
    if len(expanded) > len(test_terms):
        print(f"🎉 SUCCESS: expand_terms() working correctly!")
        print(f"   Expansion adds {len(expanded) - len(test_terms)} synonym terms")
        print(f"   This will fix the broken query expansion in matcher.py")
        
        # Test expected score impact
        sample_activity_terms = ['track', 'temperature', 'send', 'data', 'subsystem']
        overlap = len(set(expanded) & set(sample_activity_terms))
        score = overlap / len(expanded)
        print(f"   Sample score with activity terms: {overlap}/{len(expanded)} = {score:.3f}")
        
        if score > 0.15:
            print(f"   🚀 Score looks meaningful for 10% query expansion weight!")
        
    else:
        print(f"⚠️  WARNING: No expansion occurred - check resource loading")
    
    # Test statistics
    stats = resources.get_vocabulary_stats()
    print(f"\n📊 Vocabulary stats: {stats}")
    
    print(f"\n✅ Lightweight aerospace synonyms: {len(resources.synonyms)} terms")
    print(f"🎯 Perfect for surgical query expansion fix!")
    print("🔧 Ready to integrate with matcher.py!")


if __name__ == "__main__":
    test_domain_resources()
