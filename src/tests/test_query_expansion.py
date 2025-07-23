#!/usr/bin/env python3
"""
Test Query Expansion Functionality
==================================
Verify that query expansion is working correctly in the matcher
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.matching.domain_resources import DomainResources

def test_query_expansion():
    """Test the query expansion functionality."""
    
    print("🧪 Testing Query Expansion Functionality")
    print("=" * 60)
    
    # Initialize domain resources
    domain = DomainResources()
    
    # Test expand_terms method
    print("\n1️⃣ Testing expand_terms() method:")
    
    test_cases = [
        ["monitor", "control", "system"],
        ["transmit", "data"],
        ["fiber", "quality"],
        ["thermal", "temperature"]
    ]
    
    for test_terms in test_cases:
        expanded = domain.expand_terms(test_terms)
        print(f"\nInput: {test_terms}")
        print(f"Output: {expanded}")
        print(f"Expansion ratio: {len(expanded)/len(test_terms):.1f}x")
        
        # Show what was added
        added_terms = [t for t in expanded if t not in [term.lower() for term in test_terms]]
        if added_terms:
            print(f"Added: {added_terms}")
        else:
            print("❌ No expansion occurred!")
    
    # Test specific synonyms
    print("\n2️⃣ Testing specific synonym lookups:")
    
    key_terms = ["monitor", "control", "transmit", "quality", "fiber"]
    for term in key_terms:
        synonyms = domain.get_synonyms(term)
        if synonyms:
            print(f"\n'{term}' → {synonyms}")
        else:
            print(f"\n'{term}' → ❌ No synonyms found")
    
    # Test the actual query expansion as used in matcher
    print("\n3️⃣ Simulating matcher's expand_query_aerospace():")
    
    # Simulate a requirement and activity
    req_terms = ["monitor", "fiber", "quality", "optical", "properties"]
    act_terms = ["measure", "fiber", "diameter", "sensor"]
    
    print(f"\nRequirement terms: {req_terms}")
    print(f"Activity terms: {act_terms}")
    
    # Expand activity terms (as done in the matcher)
    expanded_activity = set(act_terms)
    for term in act_terms:
        synonyms = domain.get_synonyms(term.lower())
        if synonyms:
            expanded_activity.update(synonyms)
            print(f"  Expanded '{term}' with: {synonyms}")
    
    print(f"\nExpanded activity terms: {list(expanded_activity)}")
    
    # Calculate overlap
    req_set = set(t.lower() for t in req_terms)
    overlap = req_set & expanded_activity
    score = len(overlap) / len(req_set) if req_set else 0
    
    print(f"\nOverlap: {overlap}")
    print(f"Score: {score:.3f} ({len(overlap)}/{len(req_set)} requirement terms matched)")
    
    # Diagnostic summary
    print("\n📊 Diagnostic Summary:")
    
    stats = domain.get_vocabulary_stats()
    print(f"Total synonyms: {stats['total_synonyms']}")
    print(f"Total abbreviations: {stats['total_abbreviations']}")
    print(f"Total domain terms: {stats['total_terms']}")
    
    # Check if query expansion is likely to work
    if stats['total_synonyms'] < 10:
        print("\n❌ CRITICAL: Synonym dictionary is too small!")
        print("   Query expansion cannot work effectively.")
    elif stats['total_synonyms'] < 50:
        print("\n⚠️ WARNING: Limited synonym coverage")
        print("   Query expansion will have minimal impact.")
    else:
        print("\n✅ Synonym dictionary appears adequate")
    
    return stats

if __name__ == "__main__":
    test_query_expansion()