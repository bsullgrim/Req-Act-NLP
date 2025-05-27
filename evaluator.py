import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Set, Optional
import logging
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import re
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MatchingEvaluator:
    """
    Comprehensive evaluation framework for text matching systems using ground truth data.
    Implements standard Information Retrieval metrics: MRR, Precision@k, Recall@k, F1@k, NDCG@k
    """
    
    def __init__(self, ground_truth_file: str):
        """
        Initialize evaluator with ground truth mappings.
        
        Expected ground truth CSV format:
        - requirement_id: Unique identifier for requirement
        - activity_name: Name of the correctly matched activity
        - relevance_score: Optional relevance score (1-5, default=1 for binary)
        """
        self.ground_truth = self.load_ground_truth(ground_truth_file)
        self.evaluation_results = {}
        
    def load_ground_truth(self, file_path: str) -> Dict[str, List[Dict]]:
        """Load and parse ground truth data from manual_matches.csv style."""
        try:
            df = pd.read_csv(file_path)
            required_cols = ['ID', 'Satisfied By']
            
            if not all(col in df.columns for col in required_cols):
                raise ValueError(f"Ground truth file must contain columns: {required_cols}")
            
            # Normalize and expand Satisfied By
            def normalize_activity_name(name):
                return re.sub(r'\(context.*?\)', '', name, flags=re.IGNORECASE).strip().lower()
            
            ground_truth = defaultdict(list)
            for _, row in df.iterrows():
                req_id = str(row['ID']).strip()
                satisfied_by = row['Satisfied By']
                
                if pd.isna(satisfied_by):
                    continue
                
                activity_names = [normalize_activity_name(s) for s in satisfied_by.split(',') if s.strip()]
                
                for activity in activity_names:
                    ground_truth[req_id].append({
                        'activity_name': activity,
                        'relevance_score': 1  # default binary relevance
                    })
            
            logger.info(f"Loaded ground truth for {len(ground_truth)} requirements")
            return dict(ground_truth)
        
        except Exception as e:
            logger.error(f"Error loading ground truth: {e}")
            raise
    
    def compute_precision_at_k(self, predicted: List[str], relevant: Set[str], k: int) -> float:
        """Compute Precision@k: fraction of top-k predictions that are relevant."""
        if k <= 0:
            return 0.0
        
        top_k = predicted[:k]
        relevant_in_topk = sum(1 for item in top_k if item in relevant)
        return relevant_in_topk / min(k, len(top_k))
    
    def compute_recall_at_k(self, predicted: List[str], relevant: Set[str], k: int) -> float:
        """Compute Recall@k: fraction of relevant items found in top-k predictions."""
        if not relevant:
            return 1.0  # No relevant items to find
        
        top_k = predicted[:k]
        relevant_in_topk = sum(1 for item in top_k if item in relevant)
        return relevant_in_topk / len(relevant)
    
    def compute_f1_at_k(self, predicted: List[str], relevant: Set[str], k: int) -> float:
        """Compute F1@k: harmonic mean of Precision@k and Recall@k."""
        precision = self.compute_precision_at_k(predicted, relevant, k)
        recall = self.compute_recall_at_k(predicted, relevant, k)
        
        if precision + recall == 0:
            return 0.0
        
        return 2 * (precision * recall) / (precision + recall)
    
    def compute_reciprocal_rank(self, predicted: List[str], relevant: Set[str]) -> float:
        """Compute reciprocal rank: 1/rank of first relevant item."""
        for i, item in enumerate(predicted, 1):
            if item in relevant:
                return 1.0 / i
        return 0.0
    
    def compute_ndcg_at_k(self, predicted: List[str], relevant_scores: Dict[str, float], k: int) -> float:
        """
        Compute Normalized Discounted Cumulative Gain@k.
        Accounts for graded relevance and position-based discounting.
        """
        if k <= 0:
            return 0.0
        
        # DCG@k for predicted ranking
        dcg = 0.0
        for i, item in enumerate(predicted[:k]):
            if item in relevant_scores:
                relevance = relevant_scores[item]
                dcg += (2**relevance - 1) / np.log2(i + 2)  # i+2 because log2(1) = 0
        
        # IDCG@k (ideal DCG) - perfect ranking of relevant items
        ideal_scores = sorted(relevant_scores.values(), reverse=True)[:k]
        idcg = sum((2**score - 1) / np.log2(i + 2) for i, score in enumerate(ideal_scores))
        
        return dcg / idcg if idcg > 0 else 0.0
    
    def evaluate_predictions(self, predictions_df: pd.DataFrame, 
                           k_values: List[int] = [1, 3, 5, 10]) -> Dict[str, any]:
        """
        Evaluate predictions against ground truth.
        
        Expected predictions_df format:
        - ID or requirement_id: Requirement identifier
        - Activity Name: Predicted activity name
        - Combined Score: Prediction confidence score
        """
        # Standardize column names
        id_col = 'ID' if 'ID' in predictions_df.columns else 'requirement_id'
        activity_col = 'Activity Name' if 'Activity Name' in predictions_df.columns else 'activity_name'
        score_col = 'Combined Score' if 'Combined Score' in predictions_df.columns else 'score'
        
        if not all(col in predictions_df.columns for col in [id_col, activity_col]):
            raise ValueError(f"Predictions must contain columns: {[id_col, activity_col]}")
        
        # Group predictions by requirement
        predictions_by_req = defaultdict(list)
        for _, row in predictions_df.iterrows():
            req_id = str(row[id_col])
            activity = row[activity_col]
            score = row.get(score_col, 0.0)
            predictions_by_req[req_id].append((activity, score))
        
        # Sort predictions by score (descending)
        for req_id in predictions_by_req:
            predictions_by_req[req_id].sort(key=lambda x: x[1], reverse=True)
        
        # Compute metrics
        results = {
            'total_requirements': len(self.ground_truth),
            'requirements_with_predictions': len(predictions_by_req),
            'coverage': len(predictions_by_req) / len(self.ground_truth),
            'metrics': {}
        }
        
        # Initialize metric storage
        for k in k_values:
            results['metrics'][f'precision_at_{k}'] = []
            results['metrics'][f'recall_at_{k}'] = []
            results['metrics'][f'f1_at_{k}'] = []
            results['metrics'][f'ndcg_at_{k}'] = []
        
        results['metrics']['reciprocal_rank'] = []
        
        # Per-requirement analysis
        detailed_results = []
        
        for req_id, ground_truth_items in self.ground_truth.items():
            # Extract relevant activities and their scores
            relevant_activities = {item['activity_name'] for item in ground_truth_items}
            relevant_scores = {item['activity_name']: item['relevance_score'] 
                             for item in ground_truth_items}
            
            if req_id in predictions_by_req:
                # Get predicted activities (sorted by score)
                predicted_activities = [item[0] for item in predictions_by_req[req_id]]
                
                # Compute metrics for this requirement
                rr = self.compute_reciprocal_rank(predicted_activities, relevant_activities)
                results['metrics']['reciprocal_rank'].append(rr)
                
                req_result = {
                    'requirement_id': req_id,
                    'num_relevant': len(relevant_activities),
                    'num_predicted': len(predicted_activities),
                    'reciprocal_rank': rr,
                    'relevant_found': len(set(predicted_activities) & relevant_activities)
                }
                
                for k in k_values:
                    precision_k = self.compute_precision_at_k(predicted_activities, relevant_activities, k)
                    recall_k = self.compute_recall_at_k(predicted_activities, relevant_activities, k)
                    f1_k = self.compute_f1_at_k(predicted_activities, relevant_activities, k)
                    ndcg_k = self.compute_ndcg_at_k(predicted_activities, relevant_scores, k)
                    
                    results['metrics'][f'precision_at_{k}'].append(precision_k)
                    results['metrics'][f'recall_at_{k}'].append(recall_k)
                    results['metrics'][f'f1_at_{k}'].append(f1_k)
                    results['metrics'][f'ndcg_at_{k}'].append(ndcg_k)
                    
                    req_result[f'precision_at_{k}'] = precision_k
                    req_result[f'recall_at_{k}'] = recall_k
                    req_result[f'f1_at_{k}'] = f1_k
                    req_result[f'ndcg_at_{k}'] = ndcg_k
                
                detailed_results.append(req_result)
            else:
                # No predictions for this requirement
                results['metrics']['reciprocal_rank'].append(0.0)
                for k in k_values:
                    results['metrics'][f'precision_at_{k}'].append(0.0)
                    results['metrics'][f'recall_at_{k}'].append(0.0)
                    results['metrics'][f'f1_at_{k}'].append(0.0)
                    results['metrics'][f'ndcg_at_{k}'].append(0.0)
        
        # Compute aggregate metrics
        results['aggregate_metrics'] = {}
        for metric_name, values in results['metrics'].items():
            results['aggregate_metrics'][metric_name] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'median': np.median(values),
                'min': np.min(values),
                'max': np.max(values)
            }
        
        # Add MRR (Mean Reciprocal Rank)
        results['aggregate_metrics']['MRR'] = {
            'mean': np.mean(results['metrics']['reciprocal_rank']),
            'std': np.std(results['metrics']['reciprocal_rank'])
        }
        
        # Store detailed results
        results['detailed_results'] = pd.DataFrame(detailed_results)
        
        self.evaluation_results = results
        return results
    
    def print_evaluation_summary(self):
        """Print a comprehensive evaluation summary."""
        if not self.evaluation_results:
            logger.error("No evaluation results available. Run evaluate_predictions() first.")
            return
        
        results = self.evaluation_results
        
        print(f"\n{'='*70}")
        print("MATCHING EVALUATION SUMMARY")
        print(f"{'='*70}")
        
        print(f"Dataset Coverage:")
        print(f"  Total requirements: {results['total_requirements']}")
        print(f"  Requirements with predictions: {results['requirements_with_predictions']}")
        print(f"  Coverage: {results['coverage']:.1%}")
        
        print(f"\nKey Performance Metrics:")
        agg = results['aggregate_metrics']
        print(f"  MRR (Mean Reciprocal Rank): {agg['MRR']['mean']:.3f} ± {agg['MRR']['std']:.3f}")
        
        print(f"\n  Precision@k:")
        for k in [1, 3, 5, 10]:
            if f'precision_at_{k}' in agg:
                mean_p = agg[f'precision_at_{k}']['mean']
                std_p = agg[f'precision_at_{k}']['std']
                print(f"    P@{k}: {mean_p:.3f} ± {std_p:.3f}")
        
        print(f"\n  Recall@k:")
        for k in [1, 3, 5, 10]:
            if f'recall_at_{k}' in agg:
                mean_r = agg[f'recall_at_{k}']['mean']
                std_r = agg[f'recall_at_{k}']['std']
                print(f"    R@{k}: {mean_r:.3f} ± {std_r:.3f}")
        
        print(f"\n  F1@k:")
        for k in [1, 3, 5, 10]:
            if f'f1_at_{k}' in agg:
                mean_f1 = agg[f'f1_at_{k}']['mean']
                std_f1 = agg[f'f1_at_{k}']['std']
                print(f"    F1@{k}: {mean_f1:.3f} ± {std_f1:.3f}")
        
        print(f"\n  NDCG@k:")
        for k in [1, 3, 5, 10]:
            if f'ndcg_at_{k}' in agg:
                mean_ndcg = agg[f'ndcg_at_{k}']['mean']
                std_ndcg = agg[f'ndcg_at_{k}']['std']
                print(f"    NDCG@{k}: {mean_ndcg:.3f} ± {std_ndcg:.3f}")
    
    def plot_evaluation_metrics(self, save_path: Optional[str] = None):
        """Create visualization of evaluation metrics."""
        if not self.evaluation_results:
            logger.error("No evaluation results available.")
            return
        
        agg = self.evaluation_results['aggregate_metrics']
        
        # Set up the plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Text Matching Evaluation Results', fontsize=16, fontweight='bold')
        
        # Precision@k plot
        k_values = [1, 3, 5, 10]
        precision_means = [agg[f'precision_at_{k}']['mean'] for k in k_values if f'precision_at_{k}' in agg]
        precision_stds = [agg[f'precision_at_{k}']['std'] for k in k_values if f'precision_at_{k}' in agg]
        
        axes[0, 0].bar(range(len(k_values)), precision_means, yerr=precision_stds, capsize=5, alpha=0.7)
        axes[0, 0].set_title('Precision@k')
        axes[0, 0].set_xlabel('k')
        axes[0, 0].set_ylabel('Precision')
        axes[0, 0].set_xticks(range(len(k_values)))
        axes[0, 0].set_xticklabels([f'@{k}' for k in k_values])
        axes[0, 0].grid(True, alpha=0.3)
        
        # Recall@k plot
        recall_means = [agg[f'recall_at_{k}']['mean'] for k in k_values if f'recall_at_{k}' in agg]
        recall_stds = [agg[f'recall_at_{k}']['std'] for k in k_values if f'recall_at_{k}' in agg]
        
        axes[0, 1].bar(range(len(k_values)), recall_means, yerr=recall_stds, capsize=5, alpha=0.7, color='orange')
        axes[0, 1].set_title('Recall@k')
        axes[0, 1].set_xlabel('k')
        axes[0, 1].set_ylabel('Recall')
        axes[0, 1].set_xticks(range(len(k_values)))
        axes[0, 1].set_xticklabels([f'@{k}' for k in k_values])
        axes[0, 1].grid(True, alpha=0.3)
        
        # F1@k plot
        f1_means = [agg[f'f1_at_{k}']['mean'] for k in k_values if f'f1_at_{k}' in agg]
        f1_stds = [agg[f'f1_at_{k}']['std'] for k in k_values if f'f1_at_{k}' in agg]
        
        axes[1, 0].bar(range(len(k_values)), f1_means, yerr=f1_stds, capsize=5, alpha=0.7, color='green')
        axes[1, 0].set_title('F1@k')
        axes[1, 0].set_xlabel('k')
        axes[1, 0].set_ylabel('F1 Score')
        axes[1, 0].set_xticks(range(len(k_values)))
        axes[1, 0].set_xticklabels([f'@{k}' for k in k_values])
        axes[1, 0].grid(True, alpha=0.3)
        
        # NDCG@k plot
        ndcg_means = [agg[f'ndcg_at_{k}']['mean'] for k in k_values if f'ndcg_at_{k}' in agg]
        ndcg_stds = [agg[f'ndcg_at_{k}']['std'] for k in k_values if f'ndcg_at_{k}' in agg]
        
        axes[1, 1].bar(range(len(k_values)), ndcg_means, yerr=ndcg_stds, capsize=5, alpha=0.7, color='red')
        axes[1, 1].set_title('NDCG@k')
        axes[1, 1].set_xlabel('k')
        axes[1, 1].set_ylabel('NDCG')
        axes[1, 1].set_xticks(range(len(k_values)))
        axes[1, 1].set_xticklabels([f'@{k}' for k in k_values])
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Evaluation plots saved to {save_path}")
        
        plt.show()
    
    def save_detailed_results(self, output_path: str):
        """Save detailed per-requirement results to CSV."""
        if not self.evaluation_results or 'detailed_results' not in self.evaluation_results:
            logger.error("No detailed results available.")
            return
        
        detailed_df = self.evaluation_results['detailed_results']
        detailed_df.to_csv(output_path, index=False)
        logger.info(f"Detailed results saved to {output_path}")
    
    def compare_configurations(self, config_results: Dict[str, pd.DataFrame], 
                             metric: str = 'f1_at_5') -> pd.DataFrame:
        """
        Compare multiple configuration results.
        
        Args:
            config_results: Dict mapping config names to their prediction DataFrames
            metric: Metric to use for comparison
            
        Returns:
            DataFrame with comparison results
        """
        comparison_data = []
        
        for config_name, predictions_df in config_results.items():
            logger.info(f"Evaluating configuration: {config_name}")
            results = self.evaluate_predictions(predictions_df)
            
            comparison_data.append({
                'configuration': config_name,
                'MRR': results['aggregate_metrics']['MRR']['mean'],
                'precision_at_1': results['aggregate_metrics']['precision_at_1']['mean'],
                'precision_at_5': results['aggregate_metrics']['precision_at_5']['mean'],
                'recall_at_5': results['aggregate_metrics']['recall_at_5']['mean'],
                'f1_at_5': results['aggregate_metrics']['f1_at_5']['mean'],
                'ndcg_at_5': results['aggregate_metrics']['ndcg_at_5']['mean'],
                'coverage': results['coverage']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values(metric, ascending=False)
        
        print(f"\nConfiguration Comparison (sorted by {metric}):")
        print("="*80)
        print(comparison_df.to_string(index=False, float_format='%.3f'))
        
        return comparison_df


def main():
    print("=== Running Matching Evaluation on All Configurations ===\n")

    # 1. Initialize evaluator with your custom ground truth
    evaluator = MatchingEvaluator("manual_matches.csv")

    # 2. Define paths to your matcher outputs
    config_paths = {
        'semantic': "results/semantic_focused.csv",
        'lexical': "results/lexical_focused.csv",
        'balanced': "results/balanced.csv"
    }

    config_predictions = {}
    for config_name, path in config_paths.items():
        try:
            df = pd.read_csv(path)

            # Normalize predicted activity names to match manual format
            df["Activity Name"] = df["Activity Name"].str.replace(r'\(context.*?\)', '', regex=True).str.strip().str.lower()

            config_predictions[config_name] = df
        except Exception as e:
            logger.error(f"Error loading {config_name} predictions: {e}")

    # 3. Evaluate each configuration individually
    for name, df in config_predictions.items():
        print(f"\n=== Evaluation: {name.upper()} Configuration ===")
        evaluator.evaluate_predictions(df)
        evaluator.print_evaluation_summary()
        evaluator.plot_evaluation_metrics(f"evaluation_plot_{name}.png")
        evaluator.save_detailed_results(f"evaluation_details_{name}.csv")

    # 4. Compare all configurations side-by-side
    print("\n=== Comparing Configurations ===")
    evaluator.compare_configurations(config_predictions, metric='f1_at_5')


if __name__ == "__main__":
    main()