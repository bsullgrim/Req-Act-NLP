#!/usr/bin/env python3
"""
Check if matcher.py is properly integrated with domain_resources
"""

import re
from pathlib import Path

def analyze_matcher_integration():
    """Analyze the current matcher.py integration"""
    
    print("🔍 ANALYZING MATCHER.PY INTEGRATION")
    print("=" * 50)
    
    # Read current matcher.py
    matcher_path = Path("src/matching/matcher.py")
    if not matcher_path.exists():
        matcher_path = Path("matcher.py")
    
    if not matcher_path.exists():
        print("❌ Could not find matcher.py")
        return
    
    # Read with proper encoding handling
    encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252']
    content = None
    
    for encoding in encodings:
        try:
            with open(matcher_path, 'r', encoding=encoding) as f:
                content = f.read()
            print(f"✅ Read matcher.py with encoding: {encoding}")
            break
        except UnicodeDecodeError:
            continue
    
    if content is None:
        print("❌ Could not read matcher.py with any encoding")
        return
    
    # Check key integration points
    checks = [
        ("DomainResources import", r"from.*domain_resources.*import.*DomainResources"),
        ("Domain initialization", r"self\.domain\s*=\s*DomainResources\(\)"),
        ("expand_query_aerospace method", r"def expand_query_aerospace\("),
        ("domain.expand_terms usage", r"self\.domain\.expand_terms\("),
        ("domain.get_synonyms usage", r"self\.domain\.get_synonyms\("),
        ("Hardcoded AEROSPACE_TERMS removal", r"AEROSPACE_TERMS\s*="),
    ]
    
    print("📋 Integration Checklist:")
    issues = []
    
    for check_name, pattern in checks:
        if re.search(pattern, content):
            print(f"   ✅ {check_name}")
        else:
            print(f"   ❌ {check_name}")
            issues.append(check_name)
    
    # Analyze expand_query_aerospace method specifically
    print(f"\n🔍 Analyzing expand_query_aerospace method...")
    
    # Extract the method
    method_match = re.search(
        r'def expand_query_aerospace\(.*?\n(.*?)(?=def|\Z)', 
        content, 
        re.DOTALL
    )
    
    if method_match:
        method_content = method_match.group(1)
        print(f"   📝 Method found, analyzing implementation...")
        
        # Check for proper implementation
        if "return 0.0" in method_content and "No expansion" not in method_content:
            print(f"   ❌ STILL HARDCODED TO RETURN 0.0!")
            issues.append("Hardcoded return 0.0")
        elif "self.domain.expand_terms" in method_content:
            print(f"   ✅ Uses domain.expand_terms()")
        elif "self.synonyms" in method_content:
            print(f"   ✅ Uses self.synonyms (loaded from domain)")
        else:
            print(f"   ⚠️ Implementation unclear")
            
        # Check for overlap calculation
        if "overlap" in method_content and "len(" in method_content:
            print(f"   ✅ Calculates overlap")
        else:
            print(f"   ❌ Missing overlap calculation")
            issues.append("Missing overlap calculation")
    else:
        print(f"   ❌ Method not found!")
        issues.append("Method missing")
    
    # Summary
    print(f"\n📊 Integration Summary:")
    if not issues:
        print(f"   🎉 PERFECT: All integration points verified!")
        print(f"   🚀 Ready to test performance improvement")
    else:
        print(f"   ⚠️ Issues found: {len(issues)}")
        for issue in issues:
            print(f"     - {issue}")
        
        # Provide specific fixes
        print(f"\n🔧 Required fixes:")
        if "Hardcoded return 0.0" in issues:
            print(f"   1. Replace hardcoded 'return 0.0' with actual calculation")
        if "Missing overlap calculation" in issues:
            print(f"   2. Add overlap calculation between expanded terms and activity terms")
        if "domain.expand_terms usage" in issues:
            print(f"   3. Use self.domain.expand_terms() instead of hardcoded expansion")
    
    return issues

def suggest_fixes():
    """Suggest specific fixes for common issues"""
    
    print(f"\n💡 COMMON FIXES FOR EXPAND_QUERY_AEROSPACE:")
    print(f"=" * 50)
    
    print(f"""
🔧 FIXED METHOD TEMPLATE:

def expand_query_aerospace(self, query_terms: List[str], activity_terms: List[str]) -> Tuple[float, str]:
    \"\"\"FIXED: Query expansion using aerospace synonyms with proper scoring.\"\"\"
    
    # Start with original query terms
    expanded_terms = set(query_terms)
    
    # Apply synonym expansion using domain resources
    for term in query_terms:
        term_lower = term.lower()
        synonyms = self.domain.get_synonyms(term_lower)  # USE DOMAIN RESOURCES!
        for synonym in synonyms:
            expanded_terms.add(synonym.lower())
    
    # Limit expansion to prevent noise
    expanded_terms = list(expanded_terms)[:6]  # Cap at 6 terms
    
    if not expanded_terms:
        return 0.0, "No expansion terms found"
    
    # Calculate overlap with activity terms
    activity_terms_lower = [term.lower() for term in activity_terms]
    overlap = len(set(expanded_terms) & set(activity_terms_lower))
    
    # Compute score
    score = overlap / len(expanded_terms) if expanded_terms else 0.0
    
    # Create explanation
    explanation = f"Query expansion: {{overlap}}/{{len(expanded_terms)}} matches"
    
    return score, explanation
""")

def main():
    """Run the analysis"""
    issues = analyze_matcher_integration()
    
    if issues:
        suggest_fixes()
        print(f"\n🎯 Next steps:")
        print(f"  1. Fix the issues listed above")
        print(f"  2. Run test_query_expansion.py to verify")
        print(f"  3. Run matcher and check F1@5 improvement")
    else:
        print(f"\n🎯 Integration looks good! Next steps:")
        print(f"  1. Run test_query_expansion.py to verify")
        print(f"  2. Run matcher: python src/matching/matcher.py")
        print(f"  3. Run evaluation: python simple_evaluation.py")
        print(f"  4. Verify F1@5 ≥ 0.30 (up from 0.233)")

if __name__ == "__main__":
    main()