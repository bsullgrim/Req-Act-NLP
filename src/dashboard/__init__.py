"""
Dashboard Package - Modular requirements traceability evaluation dashboard
"""
from typing import Dict, Any, List, Optional
from pathlib import Path
import pandas as pd

from .core import UnifiedEvaluationDashboard
from .data import DataProcessor
from .charts import ChartGenerator
from .tables import TableGenerator
from .templates import HTMLTemplateGenerator
from .exports import DataExporter

__version__ = "1.0.0"
__all__ = [
    "UnifiedEvaluationDashboard",
    "DataProcessor",
    "ChartGenerator",
    "TableGenerator",
    "HTMLTemplateGenerator",
    "DataExporter",
    "create_unified_dashboard"
]

def create_unified_dashboard(predictions_df: pd.DataFrame,
                           ground_truth_file: Optional[str] = None,
                           requirements_df: Optional[pd.DataFrame] = None,
                           evaluation_results: Optional[Dict] = None,
                           quality_results: Optional[Dict] = None,
                           output_dir: Optional[str] = None,
                           repo_manager=None) -> str:
    
    # Setup repository structure
    if repo_manager is None:
        raise ValueError("Repository manager is required")
    
    # Use proper dashboard directory
    if output_dir is None:
        output_dir = str(repo_manager.structure['evaluation_dashboards'])
    
    # Load ground truth if provided (only if we don't already have evaluation results)
    ground_truth = None
    if ground_truth_file and Path(ground_truth_file).exists() and not evaluation_results:
        try:
            import sys
            import os
            sys.path.append(os.path.dirname(os.path.dirname(__file__)))
            from src.evaluation.evaluator import EnhancedMatchingEvaluator
            
            evaluator = EnhancedMatchingEvaluator(ground_truth_file)
            ground_truth = evaluator.ground_truth
            print(f"✓ Loaded ground truth: {len(ground_truth)} requirements")
        except ImportError as e:
            print(f"⚠️ Could not import evaluator: {e}")
        except Exception as e:
            print(f"⚠️ Could not load ground truth: {e}")
    elif evaluation_results and 'ground_truth' in evaluation_results:
        # Extract ground truth from evaluation results if available
        ground_truth = evaluation_results.get('ground_truth')
    
    # Create unified dashboard
    dashboard = UnifiedEvaluationDashboard(
        predictions_df=predictions_df,
        ground_truth=ground_truth,
        requirements_df=requirements_df,
        quality_results=quality_results,
        evaluation_results=evaluation_results,
        output_dir=output_dir,
        repo_manager=repo_manager  # Pass it down
    )
    
    dashboard_path = dashboard.create_dashboard()
    
    # Print capability summary
    caps = dashboard.capabilities
    print(f"\n📊 Unified Dashboard Created: {dashboard_path}")
    print(f"🔧 Capabilities:")
    print(f"   ✓ Match Exploration")
    print(f"   {'✓' if caps['validation_mode'] else '○'} Algorithm Validation")
    print(f"   {'✓' if caps['quality_analysis'] else '○'} Quality Intelligence")
    print(f"   {'✓' if caps['requirements_context'] else '○'} Full Requirements Context")
    
    return dashboard_path