# src/diagnostics/diagnostic_runner.py
"""
Diagnostic tool to analyze why domain_weighted and query_expansion components
in the current matcher have zero weights and low performance.
"""

import pandas as pd
import json
import sys
from pathlib import Path
from collections import Counter
import re
from typing import Dict, List, Set

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import repository utilities
try:
    from src.utils.repository_setup import RepositoryStructureManager
    from src.utils.file_utils import SafeFileHandler


    UTILS_AVAILABLE = True
except ImportError:
    print("⚠️ Repository utils not available - using basic file handling")
    UTILS_AVAILABLE = False

class MatcherDiagnostic:
    """Diagnose the current monolithic matcher issues."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent.parent
        self.results = {}
        
        # Setup output directory
        if UTILS_AVAILABLE:
            self.repo_manager = RepositoryStructureManager("outputs")
            self.repo_manager.setup_repository_structure()
            self.output_dir = self.repo_manager.structure['evaluation_results']
        else:
            self.output_dir = Path("outputs/evaluation_results")
            self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print("🔍 DIAGNOSING CURRENT MATCHER")
        print("="*60)
        print(f"Working directory: {self.project_root}")
        print(f"Output directory: {self.output_dir}")
        print()
    
    def find_data_files(self) -> Dict[str, Path]:
        """Find the CSV data files in the repository."""
        
        # Possible locations for data files
        search_locations = [
            self.project_root,  # Root directory
            self.project_root / "data" / "raw",  # Standard data location
            self.project_root / "data",  # Data directory
        ]
        
        required_files = ["requirements.csv", "activities.csv"]
        optional_files = ["manual_matches.csv", "synonyms.json"]
        
        found_files = {}
        
        for location in search_locations:
            if not location.exists():
                continue
                
            print(f"🔍 Searching in: {location}")
            
            for filename in required_files + optional_files:
                file_path = location / filename
                if file_path.exists() and filename not in found_files:
                    found_files[filename] = file_path
                    print(f"  ✅ Found: {filename}")
        
        # Check if we have the required files
        missing_required = [f for f in required_files if f not in found_files]
        if missing_required:
            print(f"\n❌ Missing required files: {missing_required}")
            print("Please ensure requirements.csv and activities.csv are in your project directory")
            return None
        
        print(f"\n✅ Found all required data files")
        return found_files
    
    def run_comprehensive_diagnosis(self):
        """Run complete diagnosis on the current matcher."""
        
        # Find data files
        data_files = self.find_data_files()
        if not data_files:
            return None
        
        # Load data
        try:
            file_handler = SafeFileHandler()
            req_df = file_handler.safe_read_csv(data_files["requirements.csv"])
            act_df = file_handler.safe_read_csv(data_files["activities.csv"])
            print(f"✅ Loaded data: {len(req_df)} requirements, {len(act_df)} activities")
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return None
        
        # Run all diagnostic tests
        print("\n" + "="*50)
        domain_results = self.diagnose_tfidf_domain_extraction(req_df, act_df)
        
        print("\n" + "="*50) 
        expansion_results = self.diagnose_query_expansion_coverage(req_df, act_df, data_files)
        
        print("\n" + "="*50)
        vocab_results = self.diagnose_vocabulary_gaps(req_df, act_df)
        
        print("\n" + "="*50)
        text_results = self.diagnose_short_text_issues(req_df, act_df)
        
        # Generate recommendations
        print("\n" + "="*60)
        recommendations = self.generate_actionable_recommendations()
        
        print("💡 ACTIONABLE RECOMMENDATIONS:")
        print("-" * 50)
        for i, rec in enumerate(recommendations, 1):
            print(f"{i:2}. {rec}")
        
        # Save results
        all_results = {
            'diagnosis_metadata': {
                'timestamp': pd.Timestamp.now().isoformat(),
                'data_files': {k: str(v) for k, v in data_files.items()},
                'requirements_count': len(req_df),
                'activities_count': len(act_df)
            },
            'domain_extraction': domain_results,
            'query_expansion': expansion_results,
            'vocabulary_gaps': vocab_results,
            'short_text_issues': text_results,
            'recommendations': recommendations
        }
        
        # Save to outputs directory
        results_file = self.output_dir / "matcher_diagnosis.json"
        with open(results_file, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        
        # Also save a human-readable report
        report_file = self.output_dir / "diagnostic_report.txt"
        self.save_readable_report(all_results, report_file)
        
        print(f"\n📄 Results saved:")
        print(f"  • Detailed: {results_file}")
        print(f"  • Report: {report_file}")
        
        return all_results
    
    def diagnose_tfidf_domain_extraction(self, req_df, act_df):
        """Diagnose TF-IDF domain weighting issues."""
        
        print("🔍 ANALYZING TF-IDF DOMAIN EXTRACTION")
        print("(Testing why domain_weighted component scores near zero)")
        
        # Simulate current matcher's TF-IDF approach
        req_texts = req_df["Requirement Text"].tolist()
        act_texts = act_df["Activity Name"].tolist()
        all_texts = req_texts + act_texts
        
        # Document length analysis
        req_lengths = [len(str(text).split()) for text in req_texts]
        act_lengths = [len(str(text).split()) for text in act_texts]
        
        print(f"\n📏 Document Length Analysis:")
        print(f"  Requirements avg: {sum(req_lengths)/len(req_lengths):.1f} words")
        print(f"  Activities avg: {sum(act_lengths)/len(act_lengths):.1f} words")
        print(f"  Activities range: {min(act_lengths)}-{max(act_lengths)} words")
        
        # Critical issue: Activities too short for TF-IDF
        very_short = sum(1 for l in act_lengths if l <= 3)
        short_pct = very_short / len(act_lengths) * 100
        print(f"  Very short activities (≤3 words): {very_short}/{len(act_lengths)} ({short_pct:.1f}%)")
        
        # Vocabulary analysis
        all_words = []
        for text in all_texts:
            words = re.findall(r'\b[a-zA-Z]{3,}\b', str(text).lower())
            all_words.extend(words)
        
        word_freq = Counter(all_words)
        
        # Simulate domain term extraction
        domain_terms = {}
        for word, freq in word_freq.items():
            if 1 < freq < len(all_texts) * 0.8:  # Not too rare, not too common
                # Current matcher's technical term boosting
                boost = 1.0
                if any(char.isdigit() for char in word):
                    boost *= 1.5
                if len(word) > 8:
                    boost *= 1.2
                domain_terms[word] = (freq / len(all_texts)) * boost
        
        top_domain = sorted(domain_terms.items(), key=lambda x: x[1], reverse=True)[:15]
        
        print(f"\n🏷️ Top Domain Terms Found:")
        for term, score in top_domain[:8]:
            print(f"  {term:15}: {score:.4f}")
        
        # Check aerospace relevance
        aerospace_terms = self._get_aerospace_vocab()
        aerospace_found = [term for term, _ in top_domain if term in aerospace_terms]
        
        print(f"\n🛸 Aerospace Relevance:")
        print(f"  Domain terms found: {len(top_domain)}")
        print(f"  Actually aerospace: {len(aerospace_found)} ({len(aerospace_found)/max(len(top_domain),1)*100:.1f}%)")
        if aerospace_found:
            print(f"  Aerospace terms: {', '.join(aerospace_found[:5])}")
        
        # Diagnose problems
        problems = []
        if short_pct > 60:
            problems.append("❌ CRITICAL: >60% activities too short for TF-IDF")
        if len(aerospace_found) < 3:
            problems.append("❌ CRITICAL: TF-IDF not finding aerospace vocabulary")
        if len(top_domain) < 8:
            problems.append("⚠️ WARNING: Very few domain terms extracted")
        
        print(f"\n🔧 Identified Problems:")
        for problem in problems:
            print(f"  {problem}")
        
        return {
            'avg_activity_length': sum(act_lengths)/len(act_lengths),
            'short_activity_percentage': short_pct,
            'domain_terms_found': len(top_domain),
            'aerospace_terms_found': len(aerospace_found),
            'aerospace_examples': aerospace_found[:5],
            'problems': problems,
            'recommendation': "Replace TF-IDF with simple aerospace term matching" if problems else "TF-IDF should work"
        }
    
    def diagnose_query_expansion_coverage(self, req_df, act_df, data_files):
        """Diagnose query expansion issues."""
        
        print("🔍 ANALYZING QUERY EXPANSION")
        print("(Testing why query_expansion component scores near zero)")
        
        # Load synonyms
        synonyms_file = data_files.get("synonyms.json")
        if not synonyms_file:
            print("❌ No synonyms.json found - query expansion disabled")
            return {
                'error': 'No synonyms file',
                'problems': ['❌ No synonyms.json file found'],
                'recommendation': 'Create synonyms.json with aerospace vocabulary'
            }
        
        try:
            with open(synonyms_file, 'r') as f:
                synonyms = json.load(f)
            print(f"📚 Loaded synonyms: {len(synonyms)} main terms")
        except Exception as e:
            print(f"❌ Error loading synonyms: {e}")
            return {'error': f'Could not load synonyms: {e}'}
        
        # Extract vocabularies
        req_vocab = self._extract_vocab(req_df["Requirement Text"])
        act_vocab = self._extract_vocab(act_df["Activity Name"])
        
        print(f"\n📊 Vocabulary Analysis:")
        print(f"  Requirement terms: {len(req_vocab)}")
        print(f"  Activity terms: {len(act_vocab)}")
        print(f"  Direct overlap: {len(req_vocab & act_vocab)} ({len(req_vocab & act_vocab)/len(req_vocab)*100:.1f}%)")
        
        # Test expansion coverage
        expandable = [term for term in req_vocab if term in synonyms]
        coverage = len(expandable) / len(req_vocab) if req_vocab else 0
        
        print(f"\n🔄 Expansion Coverage:")
        print(f"  Expandable terms: {len(expandable)}/{len(req_vocab)} ({coverage*100:.1f}%)")
        if expandable:
            print(f"  Examples: {', '.join(expandable[:6])}")
        
        # Test expansion effectiveness
        successful_expansions = []
        total_attempts = 0
        
        for term in expandable[:15]:  # Test subset
            if term in synonyms:
                for synonym in synonyms[term][:3]:
                    total_attempts += 1
                    if synonym in act_vocab:
                        successful_expansions.append(f"{term}→{synonym}")
        
        success_rate = len(successful_expansions) / total_attempts if total_attempts > 0 else 0
        
        print(f"\n🎯 Expansion Effectiveness:")
        print(f"  Success rate: {len(successful_expansions)}/{total_attempts} ({success_rate*100:.1f}%)")
        if successful_expansions:
            print(f"  Examples: {', '.join(successful_expansions[:4])}")
        
        # Sample expansion test
        print(f"\n🧪 Sample Expansion Test:")
        sample_req = req_df["Requirement Text"].iloc[0] if len(req_df) > 0 else ""
        if sample_req:
            words = re.findall(r'\b[a-zA-Z]{4,}\b', sample_req.lower())
            expanded = []
            for word in words[:5]:
                if word in synonyms:
                    expanded.extend(synonyms[word][:2])
            
            if expanded:
                matches = [syn for syn in expanded if syn in act_vocab]
                print(f"  Sample req: {sample_req[:60]}...")
                print(f"  Expansions: {', '.join(expanded[:6])}")
                print(f"  Matches: {', '.join(matches) if matches else 'NONE'}")
            else:
                print(f"  Sample req: {sample_req[:60]}...")
                print(f"  No expansions possible")
        
        # Diagnose problems
        problems = []
        if coverage < 0.15:
            problems.append("❌ CRITICAL: <15% requirement terms can be expanded")
        if success_rate < 0.15:
            problems.append("❌ CRITICAL: Expanded terms don't match activity vocabulary")
        if len(req_vocab & act_vocab) / len(req_vocab) > 0.7:
            problems.append("ℹ️ INFO: High direct overlap - expansion less critical")
        
        print(f"\n🔧 Identified Problems:")
        for problem in problems:
            print(f"  {problem}")
        
        recommendation = "Rebuild synonyms with domain-specific terms" if problems else "Query expansion should work"
        
        return {
            'expansion_coverage': coverage,
            'success_rate': success_rate,
            'expandable_terms': len(expandable),
            'successful_examples': successful_expansions[:5],
            'problems': problems,
            'recommendation': recommendation
        }
    
    def diagnose_vocabulary_gaps(self, req_df, act_df):
        """Identify vocabulary mismatches."""
        
        print("🔍 ANALYZING VOCABULARY GAPS")
        
        req_vocab = self._extract_vocab(req_df["Requirement Text"])
        act_vocab = self._extract_vocab(act_df["Activity Name"])
        
        common = req_vocab & act_vocab
        req_only = req_vocab - act_vocab
        act_only = act_vocab - req_vocab
        
        overlap_pct = len(common) / len(req_vocab) * 100 if req_vocab else 0
        
        print(f"\n📊 Vocabulary Overlap:")
        print(f"  Total overlap: {len(common)}/{len(req_vocab)} ({overlap_pct:.1f}%)")
        print(f"  Req-only terms: {len(req_only)}")
        print(f"  Act-only terms: {len(act_only)}")
        
        # Show examples
        print(f"\n📝 Examples:")
        print(f"  Common: {', '.join(list(common)[:8])}")
        print(f"  Req-only: {', '.join(list(req_only)[:8])}")
        print(f"  Act-only: {', '.join(list(act_only)[:8])}")
        
        # Look for potential synonyms
        aerospace_vocab = self._get_aerospace_vocab()
        potential_pairs = []
        
        for req_term in list(req_only)[:15]:
            if req_term in aerospace_vocab:
                for act_term in list(act_only)[:15]:
                    if act_term in aerospace_vocab and self._might_be_synonyms(req_term, act_term):
                        potential_pairs.append((req_term, act_term))
        
        print(f"\n🔗 Potential Missing Synonyms:")
        if potential_pairs:
            for req_term, act_term in potential_pairs[:5]:
                print(f"  {req_term} ↔ {act_term}")
        else:
            print("  None identified")
        
        return {
            'overlap_percentage': overlap_pct,
            'potential_synonym_pairs': potential_pairs[:10],
            'req_only_aerospace': [t for t in req_only if t in aerospace_vocab][:10],
            'act_only_aerospace': [t for t in act_only if t in aerospace_vocab][:10]
        }
    
    def diagnose_short_text_issues(self, req_df, act_df):
        """Analyze issues with short activity texts."""
        
        print("🔍 ANALYZING SHORT TEXT ISSUES")
        
        act_texts = act_df["Activity Name"].tolist()
        lengths = [len(str(text).split()) for text in act_texts]
        
        distribution = {
            'very_short': sum(1 for l in lengths if l <= 3),
            'short': sum(1 for l in lengths if 4 <= l <= 6), 
            'medium': sum(1 for l in lengths if 7 <= l <= 10),
            'long': sum(1 for l in lengths if l > 10)
        }
        
        print(f"\n📏 Activity Length Distribution:")
        total = len(lengths)
        for category, count in distribution.items():
            pct = count/total*100
            print(f"  {category:12}: {count:3} ({pct:5.1f}%)")
        
        # Show examples
        categories = [
            ('very_short', [t for t in act_texts if len(str(t).split()) <= 3]),
            ('short', [t for t in act_texts if 4 <= len(str(t).split()) <= 6]),
            ('medium', [t for t in act_texts if 7 <= len(str(t).split()) <= 10]),
            ('long', [t for t in act_texts if len(str(t).split()) > 10])
        ]
        
        print(f"\n📝 Examples:")
        for category, examples in categories:
            if examples:
                print(f"  {category:12}: {examples[0]}")
        
        very_short_pct = distribution['very_short'] / total * 100
        problems = []
        
        if very_short_pct > 50:
            problems.append("❌ CRITICAL: >50% activities very short - algorithms will struggle")
        elif very_short_pct > 30:
            problems.append("⚠️ WARNING: >30% activities very short")
        
        print(f"\n🔧 Identified Problems:")
        for problem in problems:
            print(f"  {problem}")
        
        return {
            'length_distribution': distribution,
            'avg_length': sum(lengths)/len(lengths),
            'very_short_percentage': very_short_pct,
            'problems': problems
        }
    
    def generate_actionable_recommendations(self):
        """Generate specific, actionable recommendations."""
        
        return [
            "🎯 IMMEDIATE ACTIONS (This Week):",
            "",
            "1. DISABLE broken components:",
            "   • Set domain_weighted weight to 0.0 (it's broken)",
            "   • Set query_expansion weight to 0.0 (it's broken)",
            "   • Focus on semantic + BM25 which work",
            "",
            "2. QUICK FIXES for domain weighting:",
            "   • Replace TF-IDF with hardcoded aerospace term list",
            "   • Use simple term overlap counting",
            "   • Boost matches with aerospace vocabulary",
            "",
            "3. QUICK FIXES for query expansion:",
            "   • Create focused synonyms.json with YOUR vocabulary",
            "   • Add obvious pairs: GPS↔navigation, comm↔communication",
            "   • Test expansion on sample requirements",
            "",
            "🔧 MEDIUM-TERM FIXES (Next 2 Weeks):",
            "",
            "4. REBUILD domain component:",
            "   • Extract actual aerospace terms from your data",
            "   • Use frequency-based weighting within aerospace terms",
            "   • Test on subset before full deployment",
            "",
            "5. REBUILD query expansion:",
            "   • Mine synonyms from requirement-activity pairs",
            "   • Focus on abbreviation expansion",
            "   • Validate expansions against activity vocabulary",
            "",
            "🏗️ ARCHITECTURAL IMPROVEMENTS:",
            "",
            "6. MODULAR REFACTORING:",
            "   • Separate domain knowledge from algorithm logic",
            "   • Make vocabulary configurable/updateable",
            "   • Add component-level testing",
            "",
            "7. TESTING FRAMEWORK:",
            "   • Test each component individually",
            "   • Use this diagnostic tool to measure improvements",
            "   • Validate fixes before combining components"
        ]
    
    def save_readable_report(self, results, report_file):
        """Save human-readable diagnostic report."""
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("AEROSPACE MATCHER DIAGNOSTIC REPORT\n")
            f.write("="*50 + "\n\n")
            
            f.write(f"Generated: {results['diagnosis_metadata']['timestamp']}\n")
            f.write(f"Data files analyzed:\n")
            for name, path in results['diagnosis_metadata']['data_files'].items():
                f.write(f"  • {name}: {path}\n")
            f.write(f"Requirements: {results['diagnosis_metadata']['requirements_count']}\n")
            f.write(f"Activities: {results['diagnosis_metadata']['activities_count']}\n\n")
            
            # Domain extraction issues
            f.write("DOMAIN WEIGHTING ANALYSIS\n")
            f.write("-" * 30 + "\n")
            domain = results['domain_extraction']
            f.write(f"Average activity length: {domain['avg_activity_length']:.1f} words\n")
            f.write(f"Short activities: {domain['short_activity_percentage']:.1f}%\n")
            f.write(f"Aerospace terms found: {domain['aerospace_terms_found']}\n")
            f.write(f"Problems identified:\n")
            for problem in domain['problems']:
                f.write(f"  • {problem}\n")
            f.write(f"Recommendation: {domain['recommendation']}\n\n")
            
            # Query expansion issues
            f.write("QUERY EXPANSION ANALYSIS\n")
            f.write("-" * 30 + "\n")
            if 'error' not in results['query_expansion']:
                expansion = results['query_expansion']
                f.write(f"Expansion coverage: {expansion['expansion_coverage']*100:.1f}%\n")
                f.write(f"Success rate: {expansion['success_rate']*100:.1f}%\n")
                f.write(f"Problems identified:\n")
                for problem in expansion['problems']:
                    f.write(f"  • {problem}\n")
                f.write(f"Recommendation: {expansion['recommendation']}\n\n")
            else:
                f.write(f"Error: {results['query_expansion']['error']}\n\n")
            
            # Recommendations
            f.write("RECOMMENDATIONS\n")
            f.write("-" * 20 + "\n")
            for rec in results['recommendations']:
                f.write(f"{rec}\n")
        
        print(f"📄 Human-readable report saved to: {report_file}")
    
    def _extract_vocab(self, texts):
        """Extract vocabulary from text series."""
        vocab = set()
        for text in texts:
            words = re.findall(r'\b[a-zA-Z]{3,}\b', str(text).lower())
            vocab.update(words)
        return vocab
    
    def _get_aerospace_vocab(self):
        """Basic aerospace vocabulary for testing."""
        return {
            'satellite', 'spacecraft', 'orbit', 'attitude', 'control', 'navigation',
            'telemetry', 'command', 'communication', 'antenna', 'power', 'thermal',
            'battery', 'solar', 'sensor', 'thruster', 'propulsion', 'guidance',
            'tracking', 'monitoring', 'data', 'signal', 'transmission', 'reception',
            'ground', 'station', 'mission', 'operation', 'system', 'subsystem'
        }
    
    def _might_be_synonyms(self, term1, term2):
        """Simple heuristic for potential synonyms."""
        patterns = [
            (r'comm.*', r'communication.*'),
            (r'nav.*', r'navigation.*'),
            (r'.*sat.*', r'.*satellite.*'),
        ]
        
        for p1, p2 in patterns:
            if ((re.match(p1, term1) and re.match(p2, term2)) or
                (re.match(p2, term1) and re.match(p1, term2))):
                return True
        return False

def main():
    """Run the diagnostic tool."""
    diagnostic = MatcherDiagnostic()
    results = diagnostic.run_comprehensive_diagnosis()
    
    if results:
        print(f"\n🎯 SUMMARY:")
        print("• Review the problems identified above")
        print("• Focus on the immediate actions first")
        print("• Re-run this diagnostic after making changes")
        print("• Use the detailed JSON results for deeper analysis")

if __name__ == "__main__":
    main()