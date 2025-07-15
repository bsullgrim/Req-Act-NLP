"""
Clean Workflow Orchestrator - Focused on 3 Reports
Simple, clean workflow that produces exactly 3 outputs:
1. Matching Workbook (Excel)
2. Quality Report (Excel) 
3. Interactive Dashboard (HTML)
"""
import pandas as pd
import logging
from pathlib import Path
from typing import Dict, Optional, Any
from datetime import datetime
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['OMP_NUM_THREADS'] = '2'  # TRF can use 2 threads efficiently  
os.environ['TOKENIZERS_PARALLELISM'] = 'false'


# Import core components
from src.matching.matcher import FinalCleanMatcher
from src.evaluation.evaluator import EnhancedMatchingEvaluator
from src.utils.matching_workbook_generator import MatchingWorkbookGenerator

# Import utils for file operations
from src.utils.file_utils import SafeFileHandler
from src.utils.path_resolver import SmartPathResolver
from src.utils.repository_setup import RepositoryStructureManager

logger = logging.getLogger(__name__)

class CleanWorkflow:
    """
    Clean workflow orchestrator focused on producing 3 targeted reports.
    No redundant outputs, no complex configuration - just the essentials.
    """
    
    def __init__(self, 
                 requirements_file: str = "requirements.csv",
                 activities_file: str = "activities.csv",
                 ground_truth_file: Optional[str] = "manual_matches.csv"):
        
        # Create SINGLE repository manager for entire workflow
        self.repo_manager = RepositoryStructureManager("outputs")
        self.repo_manager.setup_repository_structure()  # Only called once
        
        # Initialize utilities with repo_manager
        self.file_handler = SafeFileHandler(repo_manager=self.repo_manager)
        self.path_resolver = SmartPathResolver(repo_manager=self.repo_manager)
        
        # Resolve file paths
        file_mapping = {
            'requirements': requirements_file,
            'activities': activities_file,
            'ground_truth': ground_truth_file
        }
        
        resolved_paths = self.path_resolver.resolve_input_files(file_mapping)
        self.requirements_file = resolved_paths['requirements']
        self.activities_file = resolved_paths['activities']
        self.ground_truth_file = resolved_paths['ground_truth']
        
        # Verify critical files exist
        self._verify_files()
        
        # Initialize components with repo_manager (ONLY ONCE)
        self.matcher = FinalCleanMatcher(repo_manager=self.repo_manager)
        
        logger.info("✓ Clean workflow initialized")
        logger.info(f"   Requirements: {self.requirements_file}")
        logger.info(f"   Activities: {self.activities_file}")
        logger.info(f"   Ground truth: {self.ground_truth_file if Path(self.ground_truth_file).exists() else 'Not available'}")

    def _verify_files(self):
        """Verify essential input files exist."""
        
        file_specs = [
            {
                'path': self.requirements_file,
                'name': 'Requirements',
                'required_columns': ['ID', 'Requirement Text'],
                'optional': False
            },
            {
                'path': self.activities_file,
                'name': 'Activities',
                'required_columns': ['Activity Name'],
                'optional': False
            },
            {
                'path': self.ground_truth_file,
                'name': 'Ground Truth',
                'required_columns': ['ID', 'Satisfied By'],
                'optional': True
            }
        ]
        
        all_valid, error_messages = self.file_handler.verify_input_files(file_specs)
        
        if not all_valid:
            missing_files = [spec['path'] for spec in file_specs 
                           if not Path(spec['path']).exists() and not spec['optional']]
            alternatives = self.path_resolver.find_file_alternatives(missing_files)
            guidance = self.path_resolver.create_file_guidance(missing_files, alternatives)
            
            logger.error("❌ Input file validation failed:")
            for error in error_messages:
                logger.error(f"   {error}")
            
            print(guidance)
            raise FileNotFoundError("Required input files missing. See guidance above.")
        
        logger.info("✓ Input files validated")

    def run_workflow(self, 
                    matching_config: Optional[Dict] = None,
                    min_similarity: float = 0.35,
                    top_matches: int = 5) -> Dict[str, str]:
        """
        Run the complete workflow to generate 3 focused reports.
        """
        
        print("🚀 Clean Requirements Traceability Workflow")
        print("=" * 50)
        print("🎯 Target: 3 Focused Reports")
        print("   1. 📊 Matching Workbook (Excel)")
        print("   2. 📋 Quality Report (Excel)")
        print("   3. 🌐 Interactive Dashboard (HTML)")
        print("=" * 50)
        
        # Default matching configuration
        if matching_config is None:
            matching_config = {
                'dense_semantic': 0.4,
                'bm25': 0.2,
                'syntactic': 0.2,
                'domain_weighted': 0.1,
                'query_expansion': 0.1
            }
        
        try:
            # Step 2: Run Matching
            print("\n🔗 Step 2: Running Requirements Matching...")
            matches_df = self._run_matching(matching_config, min_similarity, top_matches)
            
            # Step 3: Run Evaluation (includes dashboard generation)
            print("\n🧪 Step 3: Running Evaluation & Creating Dashboard...")
            evaluation_results = self._run_evaluation(matches_df)
            
            # Step 4: Generate Matching Workbook
            print("\n📊 Step 4: Creating Matching Workbook...")
            workbook_path = self._generate_matching_workbook(
                evaluation_results['enhanced_predictions'], 
                evaluation_results.get('evaluation_results')
            )
            
            # Step 5: Clean up any stray files
            print("\n🧹 Step 5: Organizing Outputs...")
            self._cleanup_outputs()
            
            # Prepare final results
            results = {
                'matching_workbook': workbook_path,
                'interactive_dashboard': evaluation_results.get('dashboard_path', 'Not available')
            }
            
            self._print_completion_summary(results)
            return results
            
        except Exception as e:
            logger.error(f"❌ Workflow failed: {e}")
            raise

    def _run_matching(self, matching_config: Dict, min_similarity: float, top_matches: int) -> pd.DataFrame:
        try:
            matches_df = self.matcher.run_final_matching(
                requirements_file=self.requirements_file,
                activities_file=self.activities_file,
                weights=matching_config,
                min_sim=min_similarity,
                top_n=top_matches,
                out_file="temp_matches",
                save_explanations=False,
                repo_manager=self.repo_manager  # Pass it down
            )              
            logger.info(f"✓ Matching completed: {len(matches_df)} matches found")
            return matches_df
            
        except Exception as e:
            logger.error(f"❌ Matching failed: {e}")
            raise

    def _run_evaluation(self, matches_df: pd.DataFrame) -> Dict[str, Any]:
        """Run evaluation and create dashboard."""
        try:
            # Load requirements for context
            requirements_df = self.file_handler.safe_read_csv(self.requirements_file)
            
            # Create evaluator with repository manager (ONLY ONCE)
            evaluator = EnhancedMatchingEvaluator(
                ground_truth_file=self.ground_truth_file if Path(self.ground_truth_file).exists() else None,
                repo_manager=self.repo_manager  # Pass repo manager
            )
            
            # Run evaluation - this creates the interactive dashboard
            evaluation_results = evaluator.evaluate_predictions(
                predictions_df=matches_df,
                requirements_df=requirements_df
            )
            
            mode = "validation" if evaluation_results.get('has_ground_truth', False) else "exploration"
            logger.info(f"✓ Evaluation completed in {mode} mode")
            logger.info(f"✓ Interactive dashboard: {evaluation_results.get('dashboard_path', 'N/A')}")
            
            return evaluation_results
            
        except Exception as e:
            logger.error(f"❌ Evaluation failed: {e}")
            raise

    def _generate_matching_workbook(self, enhanced_df: pd.DataFrame, evaluation_results: Optional[Dict]) -> str:
        """Generate comprehensive matching workbook."""
        try:
            workbook_generator = MatchingWorkbookGenerator(repo_manager=self.repo_manager)
            workbook_path = workbook_generator.create_workbook(
                enhanced_df=enhanced_df,
                evaluation_results=evaluation_results
            )
            
            logger.info(f"✓ Matching workbook created: {workbook_path}")
            return workbook_path
            
        except Exception as e:
            logger.error(f"❌ Matching workbook generation failed: {e}")
            raise

    def _cleanup_outputs(self):
        """Clean up any stray files, keeping only our 3 target reports."""
        
        # Define patterns for cleanup
        cleanup_patterns = [
            "temp_matches*",
            "final_clean_matches*",
            "enhanced_predictions*",
            "discovery_*.csv",
            "high_scoring_*.csv",
            "score_gaps*.csv",
            "evaluation_summary*.json",
            "workflow_summary*.json"
        ]
        
        try:
            self.repo_manager.cleanup_stray_files(cleanup_patterns)
            logger.info("✓ Stray files cleaned up")
        except Exception as e:
            logger.warning(f"⚠️ Cleanup warning: {e}")

    def _print_completion_summary(self, results: Dict[str, str]):
        """Print clean completion summary."""
        
        print("\n" + "🎉" + "=" * 70)
        print("🎉 CLEAN WORKFLOW COMPLETED SUCCESSFULLY!")
        print("🎉" + "=" * 70)
        
        print(f"\n🎯 3 Focused Reports Generated:")
        
        report_info = [
            ("📊 Matching Workbook", results['matching_workbook'], "Engineering teams review"),
            ("🌐 Interactive Dashboard", results['interactive_dashboard'], "Technical analysis")
        ]
        
        for icon_name, path, purpose in report_info:
            status = "✓" if path and Path(path).exists() else "❌"
            print(f"   {status} {icon_name:25}: {path}")
            print(f"      💡 Purpose: {purpose}")
        
        print(f"\n📁 All outputs in repository structure:")
        print(f"   📂 outputs/")
        print(f"   ├── 📊 engineering_review/matching_workbook.xlsx")
        print(f"   └── 🌐 evaluation_results/dashboards/evaluation_dashboard.html")
        
        print(f"\n✅ Workflow focused on business value - no redundant outputs!")

def main():
    """Main entry point for clean workflow."""
    
    import argparse
    
    parser = argparse.ArgumentParser(description="Clean Requirements Traceability Workflow")
    parser.add_argument("--requirements", default="requirements.csv",
                       help="Requirements CSV file")
    parser.add_argument("--activities", default="activities.csv", 
                       help="Activities CSV file")
    parser.add_argument("--ground-truth", default="manual_matches.csv",
                       help="Ground truth CSV file (optional)")
    parser.add_argument("--min-similarity", type=float, default=0.35,
                       help="Minimum similarity threshold")
    parser.add_argument("--top-matches", type=int, default=5,
                       help="Top N matches per requirement")
    
    args = parser.parse_args()
    
    try:
        # Initialize workflow
        workflow = CleanWorkflow(
            requirements_file=args.requirements,
            activities_file=args.activities,
            ground_truth_file=args.ground_truth
        )
        
        # Run workflow
        results = workflow.run_workflow(
            min_similarity=args.min_similarity,
            top_matches=args.top_matches
        )
        
        print(f"\n🎊 Success! Check outputs/ directory for your 3 reports.")
        
    except FileNotFoundError as e:
        print(f"\n❌ File setup required: {e}")
        print(f"\n💡 Quick fix:")
        print(f"   1. Place requirements.csv and activities.csv in current directory")
        print(f"   2. Optionally add manual_matches.csv for validation")
        print(f"   3. Re-run: python clean_workflow.py")
        
    except Exception as e:
        logger.error(f"💥 Workflow failed: {e}")
        print(f"\n❌ Critical error: {e}")
        print(f"🔧 Check logs for details")

if __name__ == "__main__":
    main()