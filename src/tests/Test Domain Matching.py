#!/usr/bin/env python3
"""
Test Domain Matcher - Verify enhanced domain scoring is working
"""

import sys
import os
from pathlib import Path

# Add project root to path - handle running from anywhere
current_file = Path(__file__).resolve()
# Go up until we find the project root (contains src folder)
project_root = current_file.parent
while not (project_root / 'src').exists() and project_root.parent != project_root:
    project_root = project_root.parent

if not (project_root / 'src').exists():
    print(f"❌ Could not find src folder. Current path: {current_file}")
    print(f"❌ Searched up to: {project_root}")
    sys.exit(1)

sys.path.insert(0, str(project_root))
print(f"✅ Using project root: {project_root}")

# Now import
try:
    from src.matching.matcher import AerospaceMatcher
    print("✅ Successfully imported AerospaceMatcher")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print(f"Python path: {sys.path}")
    sys.exit(1)

def test_domain_matching():
    """Test the enhanced domain matching with various examples."""
    
    print("🧪 Testing Enhanced Domain Matching")
    print("=" * 60)
    
    # Initialize matcher
    matcher = AerospaceMatcher()
    
    # Test cases that should benefit from domain knowledge
    test_cases = [
        {
            'req': "The system shall monitor fiber quality during production",
            'act': "Measure fiber diameter",
            'expected': 'HIGH - phrase pattern match + shared terms'
        },
        {
            'req': "The system shall maintain thermal conditions within specified limits",
            'act': "Monitor system temperature",
            'expected': 'HIGH - thermal/temperature synonym + phrase pattern'
        },
        {
            'req': "The system shall protect crew members from safety hazards",
            'act': "Detect safety hazards (Autonomous hazard detection)",
            'expected': 'HIGH - safety is strong indicator + phrase match'
        },
        {
            'req': "The system shall transmit telemetry data to ground stations",
            'act': "Send data to ground control",
            'expected': 'HIGH - transmit/send synonym + data/ground indicators'
        },
        {
            'req': "The system shall operate autonomously",
            'act': "Perform automated operations",
            'expected': 'MEDIUM - autonomous/automated relationship'
        }
    ]
    
    print("\n📊 Domain Scoring Results:\n")
    
    for i, test in enumerate(test_cases, 1):
        req = test['req']
        act = test['act']
        
        # Get domain score
        score = matcher.compute_domain_aerospace_similarity(req, act)
        
        print(f"Test {i}:")
        print(f"  Requirement: {req}")
        print(f"  Activity: {act}")
        print(f"  Domain Score: {score:.3f}")
        print(f"  Expected: {test['expected']}")
        
        # Try to explain (if method exists)
        if hasattr(matcher, 'explain_domain_match'):
            explanation = matcher.explain_domain_match(req, act)
            print(f"  Components: {explanation['components']}")
        
        print()
    
    # Test the full matching pipeline
    print("\n🔄 Testing Full Matching Pipeline:\n")
    
    req = "The system shall monitor fiber quality during production"
    activities = [
        "Measure fiber diameter",
        "Monitor fiber quality",
        "Adjust production parameters (Autonomous feedback loop)",
        "Store performance data",
        "Activate backup systems"
    ]
    
    print(f"Requirement: {req}\n")
    print("Activity Matches:")
    
    for act in activities:
        # Get all component scores
        semantic = matcher.compute_semantic_similarity(req, act)
        bm25 = matcher.compute_bm25_similarity(req, act)
        domain = matcher.compute_domain_aerospace_similarity(req, act)
        
        # Expand activity for query expansion
        act_expanded = matcher.expand_aerospace_abbreviations(act)
        act_terms = matcher._tokenize(act_expanded.lower())
        expanded_act_terms = set(matcher.domain_resources.expand_terms(list(act_terms)))
        
        req_expanded = matcher.expand_aerospace_abbreviations(req)
        req_terms = set(matcher._tokenize(req_expanded.lower()))
        
        overlap = len(req_terms & expanded_act_terms) / len(req_terms) if req_terms else 0
        query_exp = overlap
        
        # Combine (simple average for test)
        combined = (semantic + bm25 + domain + query_exp) / 4
        
        print(f"  {act}")
        print(f"    Semantic: {semantic:.3f}, BM25: {bm25:.3f}, Domain: {domain:.3f}, QE: {query_exp:.3f}")
        print(f"    Combined: {combined:.3f}")
        print()

if __name__ == "__main__":
    test_domain_matching()