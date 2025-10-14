"""
Evaluation-Only Workflow - Run evaluator and dashboard on existing match results
Skips the time-consuming matching step
"""

import pandas as pd
import logging
from pathlib import Path
from typing import Optional, Dict

# Import your existing components
from src.evaluation.evaluator import EnhancedMatchingEvaluator
from src.utils.repository_setup import RepositoryStructureManager
from src.utils.matching_workbook_generator import create_matching_workbook

logger = logging.getLogger(__name__)

def run_evaluation_only_workflow(
    matches_file: str = "outputs/matching_results/aerospace_matches.csv",
    requirements_file: str = "requirements.csv", 
    manual_matches_file: Optional[str] = "manual_matches.csv",
    create_dashboard: bool = True) -> Dict:
    """
    Run evaluation and dashboard creation on existing match results.
    """
    
    print("🚀 EVALUATION-ONLY WORKFLOW")
    print("=" * 50)
    
    # Setup repository structure
    repo_manager = RepositoryStructureManager("outputs")
    repo_manager.setup_repository_structure()
    
    # Use the existing file utilities that already work
    from src.utils.file_utils import SafeFileHandler
    from src.utils.path_resolver import SmartPathResolver
    
    file_handler = SafeFileHandler(repo_manager)
    path_resolver = SmartPathResolver(repo_manager)
    
    # 1. Load existing match results
    print(f"📂 Loading existing matches from: {matches_file}")
    try:
        matches_df = file_handler.safe_read_csv(matches_file)
        print(f"✅ Loaded {len(matches_df)} matches for {matches_df['Requirement_ID'].nunique()} requirements")
    except FileNotFoundError:
        print(f"❌ ERROR: Matches file not found: {matches_file}")
        print("   Run the matcher first to generate match results!")
        return {}
    except Exception as e:
        print(f"❌ ERROR loading matches: {e}")
        return {}
    
    # 2. Load requirements using the same logic as matcher
    requirements_df = None
    try:
        # Resolve the path using the same system as matcher
        resolved_paths = path_resolver.resolve_input_files({
            'requirements': requirements_file
        })
        requirements_path = resolved_paths['requirements']
        
        requirements_df = file_handler.safe_read_csv(requirements_path)
        print(f"✅ Loaded {len(requirements_df)} requirements from: {requirements_path}")
    except Exception as e:
        print(f"⚠️ Could not load requirements file: {e}")
        print(f"   Quality analysis will use match data only")
    
    # 3. Initialize evaluator with proper ground truth handling
    print(f"🔬 Initializing evaluator...")
    
    # Resolve manual matches path using the same system
    ground_truth_file = None
    if manual_matches_file:
        try:
            resolved_paths = path_resolver.resolve_input_files({
                'manual_matches': manual_matches_file
            })
            manual_path = resolved_paths['manual_matches']
            
            if Path(manual_path).exists():
                ground_truth_file = manual_path
                print(f"   Ground truth: {ground_truth_file}")
            else:
                print(f"   No ground truth - exploration mode")
        except Exception:
            print(f"   No ground truth - exploration mode")
    
    evaluator = EnhancedMatchingEvaluator(ground_truth_file, repo_manager)
    
    # 4. Run evaluation (includes quality analysis + dashboard)
    print(f"⚙️ Running evaluation and quality analysis...")
    results = evaluator.evaluate_predictions(matches_df, requirements_df)
    
    enhanced_predictions = results['enhanced_predictions']
    evaluation_results = results['evaluation_results']
    dashboard_path = results.get('dashboard_path', '')
    
    # 5. Create Excel workbook (primary output) - with proper repo_manager
    print(f"📊 Creating Excel workbook...")
    try:
        workbook_path = create_matching_workbook(
            enhanced_df=enhanced_predictions,
            evaluation_results=evaluation_results,
            output_path="outputs/engineering_review/traceability_analysis.xlsx",
            repo_manager=repo_manager  # Pass the repo_manager
        )
        print(f"✅ Excel workbook: {workbook_path}")
    except Exception as e:
        print(f"⚠️ Excel workbook creation failed: {e}")
        workbook_path = None
    
    # 6. Print summary
    print("\n" + "=" * 50)
    print("EVALUATION COMPLETE")
    print("=" * 50)
    print(f"📁 Enhanced predictions: outputs/evaluation_results/enhanced_predictions.csv")
    if workbook_path:
        print(f"📊 Excel workbook: {workbook_path}")
    if dashboard_path:
        print(f"🌐 Dashboard: {dashboard_path}")
    
    # Quality summary
    if 'Quality_Grade' in enhanced_predictions.columns:
        quality_dist = enhanced_predictions.groupby('Requirement_ID').first()['Quality_Grade'].value_counts()
        print(f"\n🎯 Quality Summary:")
        for grade, count in quality_dist.items():
            print(f"   {grade}: {count} requirements")
    
    # Performance summary (if ground truth available)
    if evaluation_results and 'aggregate_metrics' in evaluation_results:
        f1_score = evaluation_results['aggregate_metrics'].get('f1_at_5', {}).get('mean', 0)
        coverage = evaluation_results.get('coverage', 0)
        print(f"\n📈 Performance Summary:")
        print(f"   F1@5: {f1_score:.3f}")
        print(f"   Coverage: {coverage:.1%}")
    
    return {
        'enhanced_predictions': enhanced_predictions,
        'evaluation_results': evaluation_results,
        'workbook_path': workbook_path,
        'dashboard_path': dashboard_path,
        'success': True
    }

def main():
    """Run evaluation-only workflow from command line."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run evaluation on existing match results")
    parser.add_argument("--matches", default="outputs/matching_results/aerospace_matches.csv",
                       help="Path to existing matches CSV")
    parser.add_argument("--requirements", default="requirements.csv",
                       help="Path to requirements CSV")
    parser.add_argument("--manual", default="manual_matches.csv",
                       help="Path to manual matches CSV (optional)")
    parser.add_argument("--no-dashboard", action="store_true",
                       help="Skip dashboard creation")
    
    args = parser.parse_args()
    
    # Run workflow with the provided arguments
    # The file resolution will be handled inside the workflow using existing utilities
    results = run_evaluation_only_workflow(
        matches_file=args.matches,
        requirements_file=args.requirements,
        manual_matches_file=args.manual,
        create_dashboard=not args.no_dashboard
    )
    
    if results.get('success'):
        print(f"\n✅ Evaluation workflow completed successfully!")
        if results.get('workbook_path'):
            print(f"📊 Primary output: {results['workbook_path']}")
        if results.get('dashboard_path'):
            print(f"🌐 Dashboard: {results['dashboard_path']}")
    else:
        print(f"\n❌ Evaluation workflow failed!")
        return 1
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())