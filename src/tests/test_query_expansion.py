#!/usr/bin/env python3
"""
Test Query Expansion Fix - Verify the blocker is resolved
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from src.matching.domain_resources import DomainResources

def test_query_expansion_fix():
    """Test the specific method that was broken in matcher.py"""
    
    print("🧪 TESTING QUERY EXPANSION FIX")
    print("=" * 50)
    
    # 1. Test DomainResources loads correctly
    print("📚 Testing DomainResources loading...")
    try:
        domain = DomainResources()
        print(f"✅ DomainResources loaded successfully")
        
        # Check what was loaded
        stats = domain.get_vocabulary_stats()
        print(f"   📊 Vocabulary: {stats['total_terms']} terms")
        print(f"   🔗 Synonyms: {stats['total_synonyms']} entries") 
        print(f"   📝 Abbreviations: {stats['total_abbreviations']} entries")
        
    except Exception as e:
        print(f"❌ DomainResources failed to load: {e}")
        return False
    
    # 2. Test the KEY method that was broken
    print(f"\n🎯 Testing expand_terms() - THE CRITICAL METHOD")
    test_terms = ['monitor', 'control', 'transmit', 'system']
    
    try:
        expanded = domain.expand_terms(test_terms)
        print(f"   Input: {test_terms} ({len(test_terms)} terms)")
        print(f"   Output: {expanded} ({len(expanded)} terms)")
        print(f"   Expansion ratio: {len(expanded)/len(test_terms):.1f}x")
        
        if len(expanded) > len(test_terms):
            print(f"🎉 SUCCESS: expand_terms() is working!")
            print(f"   Added {len(expanded) - len(test_terms)} synonym terms")
        else:
            print(f"⚠️ WARNING: No expansion occurred")
            return False
            
    except Exception as e:
        print(f"❌ expand_terms() failed: {e}")
        return False
    
    # 3. Test the exact scenario from matcher.py
    print(f"\n🔧 Testing matcher.py scenario...")
    
    # Simulate what happens in expand_query_aerospace()
    query_terms = ['monitor', 'temperature', 'control']
    activity_terms = ['track', 'thermal', 'manage', 'sensor', 'data']
    
    # Get expanded terms
    expanded_terms = set(query_terms)
    for term in query_terms:
        synonyms = domain.get_synonyms(term)
        expanded_terms.update(synonyms)
    
    expanded_list = list(expanded_terms)
    
    # Calculate overlap (what the matcher does)
    activity_terms_lower = [term.lower() for term in activity_terms]
    overlap = len(set(expanded_list) & set(activity_terms_lower))
    score = overlap / len(expanded_list) if expanded_list else 0.0
    
    print(f"   Query terms: {query_terms}")
    print(f"   Expanded to: {expanded_list}")
    print(f"   Activity terms: {activity_terms}")
    print(f"   Overlap: {overlap} terms")
    print(f"   Score: {score:.3f}")
    
    if score > 0.0:
        print(f"🚀 EXCELLENT: Query expansion will contribute meaningful score!")
        print(f"   With 10% weight, this adds {score * 0.1:.3f} to combined score")
    else:
        print(f"❌ PROBLEM: Score is still 0.0")
        return False
    
    # 4. Test abbreviation expansion
    print(f"\n📝 Testing abbreviation expansion...")
    test_abbrevs = ['acs', 'eps', 'tcs']
    for abbrev in test_abbrevs:
        expansion = domain.get_abbreviation(abbrev)
        print(f"   {abbrev} → {expansion}")
    
    return True

def main():
    """Run the test"""
    success = test_query_expansion_fix()
    
    if success:
        print(f"\n✅ QUERY EXPANSION FIX VERIFIED!")
        print(f"🎯 Ready to run matcher and see F1@5 improvement")
        print(f"📈 Expected F1@5: 0.30+ (up from 0.233)")
        print(f"\nNext steps:")
        print(f"  1. Run: python src/matching/matcher.py")
        print(f"  2. Run: python simple_evaluation.py")
        print(f"  3. Verify F1@5 ≥ 0.30")
    else:
        print(f"\n❌ QUERY EXPANSION STILL BROKEN")
        print(f"🔧 Need to debug domain resource loading")

if __name__ == "__main__":
    main()