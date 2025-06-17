"""
Requirements Traceability Evaluation Package

Comprehensive evaluation framework for requirements-to-activities matching algorithms.
Supports both validation mode (with ground truth) and exploration mode (without ground truth).
Always includes quality analysis and generates interactive dashboards.

Key Components:
- EnhancedMatchingEvaluator: Main evaluation engine with adaptive architecture
- Performance metrics calculation (Precision@K, Recall@K, F1@K, MRR, NDCG)
- Discovery analysis for finding missed connections
- Quality intelligence integration
- Interactive dashboard generation

Evaluation Modes:
- Validation Mode: Full performance metrics against manual traces
- Exploration Mode: Quality analysis and coverage assessment
- Hybrid Mode: Partial ground truth with enhanced analysis
"""

from .evaluator import EnhancedMatchingEvaluator

__version__ = "2.0.0"
__author__ = "Requirements Evaluation Team"

# Export main classes
__all__ = [
    "EnhancedMatchingEvaluator",
    "evaluate_predictions",
    "run_evaluation",
    "create_evaluation_dashboard"
]

def evaluate_predictions(predictions_df, ground_truth_file=None, 
                        requirements_df=None, output_dir="outputs/evaluation_results",
                        repo_manager=None):
    """
    Convenience function for comprehensive prediction evaluation.
    
    Args:
        predictions_df: DataFrame with algorithm predictions
        ground_truth_file: Path to manual traces CSV (optional)
        requirements_df: DataFrame with requirement context (optional)
        output_dir: Directory for outputs
        repo_manager: Repository structure manager
        
    Returns:
        Dict: Evaluation results with enhanced predictions and dashboard path
        
    Example:
        >>> from src.evaluation import evaluate_predictions
        >>> results = evaluate_predictions(predictions_df, "manual_matches.csv")
        >>> print(f"Dashboard: {results['dashboard_path']}")
    """
    evaluator = EnhancedMatchingEvaluator(
        ground_truth_file=ground_truth_file,
        repo_manager=repo_manager
    )
    
    return evaluator.evaluate_predictions(predictions_df, requirements_df)

def run_evaluation(predictions_file, ground_truth_file=None, 
                  requirements_file=None, repo_manager=None):
    """
    Run complete evaluation pipeline from file inputs.
    
    Args:
        predictions_file: Path to predictions CSV
        ground_truth_file: Path to manual traces CSV (optional)
        requirements_file: Path to requirements CSV (optional)
        repo_manager: Repository structure manager
        
    Returns:
        Dict: Complete evaluation results
    """
    import pandas as pd
    
    # Load predictions
    evaluator = EnhancedMatchingEvaluator(
        ground_truth_file=ground_truth_file,
        repo_manager=repo_manager
    )
    predictions_df = evaluator._safe_read_csv(predictions_file)
    
    # Load requirements if provided
    requirements_df = None
    if requirements_file:
        requirements_df = evaluator._safe_read_csv(requirements_file)
    
    return evaluator.evaluate_predictions(predictions_df, requirements_df)

def create_evaluation_dashboard(predictions_df, evaluation_results=None,
                              requirements_df=None, output_dir=None,
                              repo_manager=None):
    """
    Create interactive evaluation dashboard.
    
    Args:
        predictions_df: Enhanced predictions DataFrame
        evaluation_results: Evaluation metrics and analysis
        requirements_df: Requirements context (optional)
        output_dir: Custom output directory
        repo_manager: Repository structure manager
        
    Returns:
        str: Path to created dashboard
    """
    import src.dashboard as dashboard
    
    return dashboard.create_unified_dashboard(
        predictions_df=predictions_df,
        evaluation_results=evaluation_results,
        requirements_df=requirements_df,
        output_dir=output_dir,
        repo_manager=repo_manager
    )

def get_evaluation_metrics():
    """
    Get list of supported evaluation metrics.
    
    Returns:
        Dict[str, str]: Metric names and descriptions
    """
    return {
        'precision_at_k': 'Precision at top-K predictions',
        'recall_at_k': 'Recall at top-K predictions', 
        'f1_at_k': 'F1-score at top-K predictions',
        'ndcg_at_k': 'Normalized Discounted Cumulative Gain at top-K',
        'mrr': 'Mean Reciprocal Rank',
        'coverage': 'Percentage of requirements with predictions',
        'discovery_rate': 'Rate of novel high-scoring connections found'
    }

def get_default_k_values():
    """Get default K values for evaluation metrics."""
    return [1, 3, 5, 10]

def validate_ground_truth(ground_truth_file):
    """
    Validate ground truth file format and content.
    
    Args:
        ground_truth_file: Path to ground truth CSV
        
    Returns:
        Tuple[bool, List[str]]: (is_valid, error_messages)
    """
    import pandas as pd
    from pathlib import Path
    
    errors = []
    
    if not Path(ground_truth_file).exists():
        errors.append(f"Ground truth file not found: {ground_truth_file}")
        return False, errors
    
    try:
        df = pd.read_csv(ground_truth_file)
        
        required_columns = ['ID', 'Satisfied By']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            errors.append(f"Missing required columns: {missing_columns}")
        
        if len(df) == 0:
            errors.append("Ground truth file is empty")
        
        # Check for valid requirement IDs
        if 'ID' in df.columns:
            null_ids = df['ID'].isnull().sum()
            if null_ids > 0:
                errors.append(f"Found {null_ids} null requirement IDs")
        
    except Exception as e:
        errors.append(f"Error reading ground truth file: {e}")
    
    return len(errors) == 0, errors

def check_evaluation_compatibility(predictions_df, ground_truth_file=None):
    """
    Check compatibility between predictions and ground truth.
    
    Args:
        predictions_df: Predictions DataFrame
        ground_truth_file: Path to ground truth file (optional)
        
    Returns:
        Dict[str, Any]: Compatibility analysis results
    """
    results = {
        'predictions_valid': True,
        'ground_truth_valid': True,
        'overlap_analysis': {},
        'recommendations': []
    }
    
    # Validate predictions
    required_pred_columns = ['ID', 'Activity Name', 'Combined Score']
    missing_pred_cols = [col for col in required_pred_columns if col not in predictions_df.columns]
    
    if missing_pred_cols:
        results['predictions_valid'] = False
        results['recommendations'].append(f"Add missing prediction columns: {missing_pred_cols}")
    
    # Validate ground truth if provided
    if ground_truth_file:
        gt_valid, gt_errors = validate_ground_truth(ground_truth_file)
        results['ground_truth_valid'] = gt_valid
        
        if not gt_valid:
            results['recommendations'].extend(gt_errors)
        else:
            # Analyze overlap
            import pandas as pd
            gt_df = pd.read_csv(ground_truth_file)
            
            pred_reqs = set(predictions_df['ID'].unique())
            gt_reqs = set(gt_df['ID'].unique())
            
            overlap = pred_reqs & gt_reqs
            pred_only = pred_reqs - gt_reqs
            gt_only = gt_reqs - pred_reqs
            
            results['overlap_analysis'] = {
                'total_overlap': len(overlap),
                'pred_only': len(pred_only),
                'gt_only': len(gt_only),
                'overlap_percentage': len(overlap) / len(pred_reqs | gt_reqs) * 100
            }
            
            if len(pred_only) > 0:
                results['recommendations'].append(f"{len(pred_only)} requirements in predictions but not ground truth")
            
            if len(gt_only) > 0:
                results['recommendations'].append(f"{len(gt_only)} requirements in ground truth but not predictions")
    
    return results

# Default evaluation configuration
DEFAULT_EVALUATION_CONFIG = {
    'k_values': [1, 3, 5, 10],
    'min_score_threshold': 0.35,
    'discovery_threshold': 0.5,
    'quality_analysis': True,
    'create_dashboard': True,
    'export_detailed_results': True
}

def get_default_evaluation_config():
    """Get default evaluation configuration."""
    return DEFAULT_EVALUATION_CONFIG.copy()

# Evaluation best practices
EVALUATION_BEST_PRACTICES = """
Requirements Traceability Evaluation Best Practices:

1. Data Preparation:
   - Ensure consistent requirement IDs across datasets
   - Validate ground truth completeness and accuracy
   - Clean and normalize activity names
   - Remove duplicate or invalid entries

2. Metric Selection:
   - Use F1@5 as primary performance indicator
   - Include MRR for ranking quality assessment
   - Monitor coverage for completeness
   - Track discovery rate for innovation

3. Validation Strategy:
   - Split ground truth for training/validation if possible
   - Use cross-validation for robust results
   - Compare against baseline methods
   - Validate with domain experts

4. Analysis Focus:
   - Investigate high-scoring discoveries
   - Analyze score distribution patterns
   - Review quality correlation with performance
   - Identify systematic biases or gaps

5. Reporting:
   - Include confidence intervals for metrics
   - Provide actionable recommendations
   - Document methodology and limitations
   - Share interactive dashboards with stakeholders
"""

def print_evaluation_best_practices():
    """Print evaluation best practices."""
    print(EVALUATION_BEST_PRACTICES)
