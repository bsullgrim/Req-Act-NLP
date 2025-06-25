"""
Enhanced Matching Evaluator - Adaptive architecture for requirements traceability evaluation
Handles cases with/without ground truth, integrates quality analysis, creates dashboard

Architecture:
- Adaptive to ground truth availability
- Always adds quality analysis 
- Always creates dashboard (content adapts to available data)
- Integrates with modular dashboard system
"""

import pandas as pd
import numpy as np
import json
import re
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict, Counter
import chardet

# Import components
from ..quality import RequirementAnalyzer, QualityMetrics

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedMatchingEvaluator:
    """Enhanced evaluator that adapts to ground truth availability and always includes quality analysis."""
    
    def __init__(self, ground_truth_file: Optional[str] = None, repo_manager=None):
        """
        Initialize evaluator with adaptive ground truth handling.
        
        Args:
            ground_truth_file: Path to manual traces CSV, or None if not available
            repo_manager: Repository structure manager, or None to create one
        """
        self.ground_truth_file = ground_truth_file
        self.ground_truth = {}
        self.has_ground_truth = False
        
        # Setup repository manager
        if repo_manager is None:
            raise ValueError("Repository manager is required")
        self.repo_manager = repo_manager
        
        # Initialize quality analyzer
        self.quality_analyzer = RequirementAnalyzer()
        
        # Load ground truth if available
        if ground_truth_file and Path(ground_truth_file).exists():
            try:
                self.ground_truth = self._load_ground_truth(ground_truth_file)
                self.has_ground_truth = True
                logger.info(f"✅ Loaded ground truth: {len(self.ground_truth)} requirements")
            except Exception as e:
                logger.warning(f"⚠️ Could not load ground truth: {e}")
                self.has_ground_truth = False
        else:
            logger.info(f"🔍 No ground truth provided - running in exploration mode")
    
    def _detect_encoding(self, file_path: str) -> str:
        """Detect file encoding with fallback."""
        try:
            with open(file_path, 'rb') as f:
                raw_data = f.read(10000)
                result = chardet.detect(raw_data)
                return result['encoding'] or 'utf-8'
        except Exception:
            return 'utf-8'
    
    def _safe_read_csv(self, file_path: str, **kwargs) -> pd.DataFrame:
        """Safely read CSV with automatic encoding detection."""
        encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252', 'iso-8859-1']
        
        # Try detected encoding first
        detected = self._detect_encoding(file_path)
        if detected not in encodings:
            encodings.insert(0, detected)
        
        for encoding in encodings:
            try:
                df = pd.read_csv(file_path, encoding=encoding, **kwargs)
                logger.info(f"Read {Path(file_path).name} with {encoding}")
                return df
            except UnicodeDecodeError:
                continue
        
        raise RuntimeError(f"Could not read {file_path} with any encoding")
    
    def _load_ground_truth(self, ground_truth_file: str) -> Dict[str, List[Dict]]:
        """Load ground truth manual traces."""
        
        df = self._safe_read_csv(ground_truth_file)
        
        # Normalize function
        def normalize_text(text):
            if pd.isna(text):
                return ""
            text = str(text).strip()
            # Remove leading numbers (e.g., "1.2.3 Activity Name" -> "Activity Name")
            text = re.sub(r'^\d+(\.\d+)*\s+', '', text)
            # Remove context information in parentheses
            text = text.split('(context')[0]
            return text.strip().lower().replace("  ", " ").replace("-", " ")
        
        ground_truth = defaultdict(list)
        
        for _, row in df.iterrows():
            req_id = str(row['ID']).strip()
            satisfied_by = str(row.get('Satisfied By', '')).strip()
            
            if satisfied_by and satisfied_by.lower() != 'nan':
                # Split multiple activities
                activities = [activity.strip() for activity in satisfied_by.split(',')]
                
                for activity in activities:
                    if activity:
                        normalized_activity = normalize_text(activity)
                        if normalized_activity:
                            ground_truth[req_id].append({
                                'activity_name': normalized_activity,
                                'original_activity': activity
                            })
        
        logger.info(f"Loaded ground truth for {len(ground_truth)} requirements")
        return dict(ground_truth)
    
    def _add_quality_analysis(self, predictions_df: pd.DataFrame, 
                            requirements_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Add comprehensive quality analysis to predictions."""
        
        logger.info("🔍 Running quality analysis on requirements...")
        
        enhanced_df = predictions_df.copy()
        
        # Get unique requirements for quality analysis
        unique_requirements = {}
        # Comprehensive column detection
        possible_req_id_cols = ['ID', 'Requirement_ID', 'requirement_id', 'req_id']
        possible_req_text_cols = ['Requirement Text', 'Requirement_Text', 'requirement_text', 'Text']

        req_id_col = None
        req_text_col = None

        for col in possible_req_id_cols:
            if col in enhanced_df.columns:
                req_id_col = col
                break

        for col in possible_req_text_cols:
            if col in enhanced_df.columns:
                req_text_col = col
                break

        if req_id_col is None:
            raise ValueError(f"Could not find requirement ID column. Available: {enhanced_df.columns.tolist()}")
        if req_text_col is None:
            raise ValueError(f"Could not find requirement text column. Available: {enhanced_df.columns.tolist()}")

        print(f"🔍 Using columns: req_id='{req_id_col}', req_text='{req_text_col}'")
        # Extract unique requirements from predictions    
        for _, row in enhanced_df.iterrows():
            req_id = str(row[req_id_col])
            req_text = str(row.get(req_text_col, ''))
            if req_id not in unique_requirements and req_text.strip():
                unique_requirements[req_id] = req_text
        
        # Store unique requirement count for dashboard metadata
        self.unique_requirements_count = len(unique_requirements)
        
        logger.info(f"Analyzing quality for {len(unique_requirements)} unique requirements...")
        
        # Run quality analysis for each unique requirement
        quality_results = {}
        for req_id, req_text in unique_requirements.items():
            try:
                issues, metrics = self.quality_analyzer.analyze_requirement(req_text, req_id)
                
                # Calculate overall quality score
                overall_quality = (
                    metrics.clarity_score * 0.25 +
                    metrics.completeness_score * 0.25 +
                    metrics.verifiability_score * 0.2 +
                    metrics.atomicity_score * 0.2 +
                    metrics.consistency_score * 0.1
                )
                
                # Determine quality grade
                if overall_quality >= 80:
                    grade = "EXCELLENT"
                elif overall_quality >= 65:
                    grade = "GOOD"
                elif overall_quality >= 50:
                    grade = "FAIR"
                elif overall_quality >= 35:
                    grade = "POOR"
                else:
                    grade = "CRITICAL"
                
                quality_results[req_id] = {
                    'overall_score': overall_quality,
                    'grade': grade,
                    'clarity': metrics.clarity_score,
                    'completeness': metrics.completeness_score,
                    'verifiability': metrics.verifiability_score,
                    'atomicity': metrics.atomicity_score,
                    'consistency': metrics.consistency_score,
                    'issues': issues,
                    'issue_count': len(issues),
                    'severity_breakdown': metrics.severity_breakdown
                }
                
            except Exception as e:
                logger.warning(f"Quality analysis failed for {req_id}: {e}")
                quality_results[req_id] = {
                    'overall_score': 50.0,
                    'grade': "UNKNOWN",
                    'clarity': 50.0,
                    'completeness': 50.0,
                    'verifiability': 50.0,
                    'atomicity': 50.0,
                    'consistency': 50.0,
                    'issues': [],
                    'issue_count': 0
                }
        
        # Store quality results for dashboard use
        self.quality_results_by_requirement = quality_results
        
        # Add quality columns to enhanced_df
        quality_columns = {
            'Quality_Score': [],
            'Quality_Grade': [],
            'Clarity_Score': [],
            'Completeness_Score': [],
            'Verifiability_Score': [],
            'Atomicity_Score': [],
            'Consistency_Score': [],
            'Quality_Issues': [],
            'Issues': [],
            'Quality_Issue_Count': [],
            'Critical_Issues': [],
            'High_Issues': [],
            'Medium_Issues': [],
            'Low_Issues': []
        }
        
        for _, row in enhanced_df.iterrows():
            req_id = str(row[req_id_col])
            quality_data = quality_results.get(req_id, quality_results.get(list(quality_results.keys())[0], {}))
            
            quality_columns['Quality_Score'].append(quality_data.get('overall_score', 50.0))
            quality_columns['Quality_Grade'].append(quality_data.get('grade', 'UNKNOWN'))
            quality_columns['Clarity_Score'].append(quality_data.get('clarity', 50.0))
            quality_columns['Completeness_Score'].append(quality_data.get('completeness', 50.0))
            quality_columns['Verifiability_Score'].append(quality_data.get('verifiability', 50.0))
            quality_columns['Atomicity_Score'].append(quality_data.get('atomicity', 50.0))
            quality_columns['Consistency_Score'].append(quality_data.get('consistency', 50.0))
            quality_columns['Quality_Issues'].append('; '.join(quality_data.get('issues', [])[:3]))
            quality_columns['Issues'].append(quality_data.get('issues', []))
            quality_columns['Quality_Issue_Count'].append(quality_data.get('issue_count', 0))
            quality_columns['Critical_Issues'].append(quality_data.get('severity_breakdown', {}).get('critical', 0))
            quality_columns['High_Issues'].append(quality_data.get('severity_breakdown', {}).get('high', 0))
            quality_columns['Medium_Issues'].append(quality_data.get('severity_breakdown', {}).get('medium', 0))
            quality_columns['Low_Issues'].append(quality_data.get('severity_breakdown', {}).get('low', 0))
        
        # Add all quality columns to enhanced_df
        for col_name, col_data in quality_columns.items():
            enhanced_df[col_name] = col_data
        
        logger.info(f"✅ Quality analysis complete. Added quality columns to {len(enhanced_df)} predictions.")
        
        
        return enhanced_df
    
    def _compute_evaluation_metrics(self, enhanced_df: pd.DataFrame) -> Dict[str, Any]:
        """Compute evaluation metrics against ground truth with FIXED manual matches."""
        
        if not self.has_ground_truth:
            return {}
        
        logger.info("📊 Computing evaluation metrics against ground truth...")
        
        # Normalize function
        def normalize_text(text):
            if pd.isna(text):
                return ""
            text = str(text).strip()
            text = re.sub(r'^\d+(\.\d+)*\s+', '', text)
            text = text.split('(context')[0]
            return text.strip().lower().replace("  ", " ").replace("-", " ")
        
        # Group predictions by requirement
        req_predictions = defaultdict(list)
        # Comprehensive column detection
        possible_req_id_cols = ['ID', 'Requirement_ID', 'requirement_id']
        possible_activity_cols = ['Activity Name', 'Activity_Name', 'activity_name']
        possible_score_cols = ['Combined Score', 'Combined_Score', 'score']

        req_id_col = next((col for col in possible_req_id_cols if col in enhanced_df.columns), None)
        activity_col = next((col for col in possible_activity_cols if col in enhanced_df.columns), None)
        score_col = next((col for col in possible_score_cols if col in enhanced_df.columns), None)

        if not all([req_id_col, activity_col, score_col]):
            raise ValueError(f"Missing columns. Found: req_id='{req_id_col}', activity='{activity_col}', score='{score_col}'. Available: {enhanced_df.columns.tolist()}")
        for _, row in enhanced_df.iterrows():
            req_id = str(row[req_id_col])
            activity = normalize_text(row[activity_col])
            score = row[score_col]
            
            req_predictions[req_id].append({
                'activity': activity,
                'score': score,
                'original_activity': row[activity_col]
            })
        
        # Sort predictions by score for each requirement
        for req_id in req_predictions:
            req_predictions[req_id].sort(key=lambda x: x['score'], reverse=True)
        
        # COMPUTE METRICS WITH ENHANCED DISCOVERY DATA
        metrics = {}
        k_values = [1, 3, 5, 10]
        
        all_precision_at_k = {k: [] for k in k_values}
        all_recall_at_k = {k: [] for k in k_values}
        all_f1_at_k = {k: [] for k in k_values}
        all_ndcg_at_k = {k: [] for k in k_values}
        all_mrr = []
        
        total_requirements = 0
        covered_requirements = 0
        
        # ENHANCED Discovery analysis with proper manual matches
        high_scoring_misses = []
        score_gaps = []
        
        for req_id in self.ground_truth:
            total_requirements += 1
            
            # Get ground truth activities for this requirement
            gt_activities = set(item['activity_name'] for item in self.ground_truth[req_id])
            gt_original_activities = [item['original_activity'] for item in self.ground_truth[req_id]]
            
            # Get predictions for this requirement
            predictions = req_predictions.get(req_id, [])
            
            if predictions:
                covered_requirements += 1
                
                # Compute standard metrics...
                for k in k_values:
                    top_k_activities = set(pred['activity'] for pred in predictions[:k])
                    
                    if top_k_activities:
                        precision_k = len(top_k_activities & gt_activities) / len(top_k_activities)
                    else:
                        precision_k = 0.0
                    all_precision_at_k[k].append(precision_k)
                    
                    if gt_activities:
                        recall_k = len(top_k_activities & gt_activities) / len(gt_activities)
                    else:
                        recall_k = 0.0
                    all_recall_at_k[k].append(recall_k)
                    
                    if precision_k + recall_k > 0:
                        f1_k = 2 * precision_k * recall_k / (precision_k + recall_k)
                    else:
                        f1_k = 0.0
                    all_f1_at_k[k].append(f1_k)
                    
                    # NDCG@k
                    dcg = 0.0
                    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(k, len(gt_activities))))
                    
                    for i, pred in enumerate(predictions[:k]):
                        if pred['activity'] in gt_activities:
                            dcg += 1.0 / np.log2(i + 2)
                    
                    ndcg_k = dcg / idcg if idcg > 0 else 0.0
                    all_ndcg_at_k[k].append(ndcg_k)
                
                # MRR
                mrr_score = 0.0
                for i, pred in enumerate(predictions):
                    if pred['activity'] in gt_activities:
                        mrr_score = 1.0 / (i + 1)
                        break
                all_mrr.append(mrr_score)
                
                # ENHANCED Discovery analysis - find high-scoring misses WITH MANUAL MATCHES
                for pred in predictions:
                    if pred['activity'] not in gt_activities and pred['score'] > 0.5:
                        high_scoring_misses.append({
                            'Requirement_ID': req_id,
                            'activity_name': pred['original_activity'],
                            'score': pred['score'],
                            'manual_matches_count': len(gt_activities),  # CORRECT: count of manual matches
                            'manual_matches': gt_original_activities,     # ENHANCED: actual manual matches list
                            'ground_truth_activities': gt_original_activities  # BACKUP: ensure we have the data
                        })
                
                # ENHANCED Score gaps analysis WITH MANUAL MATCHES
                if predictions and gt_activities:
                    max_miss_score = max((pred['score'] for pred in predictions 
                                        if pred['activity'] not in gt_activities), default=0)
                    min_manual_score = min((pred['score'] for pred in predictions 
                                        if pred['activity'] in gt_activities), default=float('inf'))
                    
                    if max_miss_score > min_manual_score and min_manual_score != float('inf'):
                        gap = max_miss_score - min_manual_score
                        
                        # Find the specific activity that caused the gap
                        max_miss_activity = next(pred['original_activity'] for pred in predictions 
                                            if pred['activity'] not in gt_activities and pred['score'] == max_miss_score)
                        
                        score_gaps.append({
                            'Requirement_ID': req_id,
                            'max_miss_score': max_miss_score,
                            'min_manual_score': min_manual_score,
                            'gap': gap,
                            'max_miss_activity': max_miss_activity,
                            'manual_matches': gt_original_activities,     # ENHANCED: include manual matches
                            'manual_matches_count': len(gt_activities)    # CORRECT: count manual matches
                        })
        
        # Aggregate metrics (unchanged)
        aggregate_metrics = {}
        for k in k_values:
            aggregate_metrics[f'precision_at_{k}'] = {
                'mean': np.mean(all_precision_at_k[k]) if all_precision_at_k[k] else 0.0,
                'std': np.std(all_precision_at_k[k]) if all_precision_at_k[k] else 0.0
            }
            aggregate_metrics[f'recall_at_{k}'] = {
                'mean': np.mean(all_recall_at_k[k]) if all_recall_at_k[k] else 0.0,
                'std': np.std(all_recall_at_k[k]) if all_recall_at_k[k] else 0.0
            }
            aggregate_metrics[f'f1_at_{k}'] = {
                'mean': np.mean(all_f1_at_k[k]) if all_f1_at_k[k] else 0.0,
                'std': np.std(all_f1_at_k[k]) if all_f1_at_k[k] else 0.0
            }
            aggregate_metrics[f'ndcg_at_{k}'] = {
                'mean': np.mean(all_ndcg_at_k[k]) if all_ndcg_at_k[k] else 0.0,
                'std': np.std(all_ndcg_at_k[k]) if all_ndcg_at_k[k] else 0.0
            }
        
        aggregate_metrics['MRR'] = {
            'mean': np.mean(all_mrr) if all_mrr else 0.0,
            'std': np.std(all_mrr) if all_mrr else 0.0
        }
        
        # Coverage
        coverage = covered_requirements / total_requirements if total_requirements > 0 else 0.0
        
        # ENHANCED Discovery summary
        discovery_summary = {
            'total_high_scoring_misses': len(high_scoring_misses),
            'requirements_with_high_misses': len(set(miss['Requirement_ID'] for miss in high_scoring_misses)),
            'score_gaps_count': len(score_gaps),
            'discovery_rate': len(high_scoring_misses) / total_requirements if total_requirements > 0 else 0.0
        }
        
        # Score distribution analysis for dashboard charts
        score_distributions = self._compute_score_distributions(req_predictions)
        
        print(f"✅ ENHANCED DISCOVERY ANALYSIS:")
        print(f"  - High scoring misses: {len(high_scoring_misses)}")
        print(f"  - Score gaps: {len(score_gaps)}")
        print(f"  - Sample manual matches for first discovery: {high_scoring_misses[0]['manual_matches'][:3] if high_scoring_misses else 'None'}")
        
        evaluation_results = {
            'aggregate_metrics': aggregate_metrics,
            'coverage': coverage,
            'total_requirements': total_requirements,
            'covered_requirements': covered_requirements,
            'discovery_analysis': {
                'high_scoring_misses': high_scoring_misses,    # ENHANCED with manual matches
                'score_gaps': score_gaps,                      # ENHANCED with manual matches  
                'summary': discovery_summary
            },
            'score_distributions': score_distributions
        }
        
        logger.info(f"✅ Evaluation metrics computed. F1@5: {aggregate_metrics.get('f1_at_5', {}).get('mean', 0):.3f}")
        
        return evaluation_results    
    def _compute_score_distributions(self, req_predictions: Dict) -> Dict[str, Any]:
        """Compute score distributions for manual vs algorithm comparison chart."""
        
        manual_scores = []
        non_manual_scores = []
        
        for req_id in self.ground_truth:
            gt_activities = set(item['activity_name'] for item in self.ground_truth[req_id])
            predictions = req_predictions.get(req_id, [])
            
            for pred in predictions:
                if pred['activity'] in gt_activities:
                    manual_scores.append(pred['score'])
                else:
                    non_manual_scores.append(pred['score'])
        
        score_distributions = {}
        
        if manual_scores:
            score_distributions['manual_scores'] = {
                'scores': manual_scores,
                'count': len(manual_scores),
                'mean': np.mean(manual_scores),
                'median': np.median(manual_scores),
                'std': np.std(manual_scores)
            }
        
        if non_manual_scores:
            score_distributions['non_manual_scores'] = {
                'scores': non_manual_scores,
                'count': len(non_manual_scores), 
                'mean': np.mean(non_manual_scores),
                'median': np.median(non_manual_scores),
                'std': np.std(non_manual_scores)
            }
        
        return score_distributions
    
    def _create_dashboard(self, enhanced_df: pd.DataFrame, 
                        evaluation_results: Dict, 
                        requirements_df: Optional[pd.DataFrame] = None) -> str:
        """Create unified dashboard using modular dashboard system."""
        
        logger.info("🎨 Creating unified dashboard...")
        
        try:
            # Import dashboard package
            import src.dashboard as dashboard
            
            # Enhance evaluation_results with correct metadata
            enhanced_evaluation_results = evaluation_results.copy()
            
            # Fix total_requirements count - use actual unique requirements count
            if hasattr(self, 'unique_requirements_count'):
                enhanced_evaluation_results['total_requirements'] = self.unique_requirements_count
            elif requirements_df is not None and not requirements_df.empty:
                enhanced_evaluation_results['total_requirements'] = len(requirements_df)
            else:
                # Fallback: count unique requirements from predictions
                enhanced_evaluation_results['total_requirements'] = enhanced_df['ID'].nunique() if 'ID' in enhanced_df.columns else 0
            
            # Add quality analysis metadata with correct counts
            if hasattr(self, 'quality_results_by_requirement'):
                quality_metadata = {
                    'unique_requirements_analyzed': len(self.quality_results_by_requirement),
                    'quality_distribution_by_requirement': {},
                    'quality_scores_by_requirement': []
                }
                
                # Count quality grades by unique requirements (not matches)
                for req_id, quality_data in self.quality_results_by_requirement.items():
                    grade = quality_data['grade']
                    if grade not in quality_metadata['quality_distribution_by_requirement']:
                        quality_metadata['quality_distribution_by_requirement'][grade] = 0
                    quality_metadata['quality_distribution_by_requirement'][grade] += 1
                    quality_metadata['quality_scores_by_requirement'].append(quality_data['overall_score'])
                
                enhanced_evaluation_results['quality_metadata'] = quality_metadata
            
            # Create dashboard with enhanced data
            dashboard_path = dashboard.create_unified_dashboard(
                predictions_df=enhanced_df,
                ground_truth_file=self.ground_truth_file if self.has_ground_truth else None,
                requirements_df=requirements_df,
                evaluation_results=enhanced_evaluation_results,
                quality_results=None,
                repo_manager=self.repo_manager  # Pass the repo manager if available
            )
            
            logger.info(f"✅ Dashboard created: {dashboard_path}")
            return dashboard_path
            
        except ImportError as e:
            logger.error(f"❌ Could not import dashboard package: {e}")
            return ""
        except Exception as e:
            logger.error(f"❌ Dashboard creation failed: {e}")
            return ""
    
    def evaluate_predictions(self, predictions_df: pd.DataFrame, 
                           requirements_df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Main evaluation method - always adds quality analysis and creates dashboard.
        
        Args:
            predictions_df: Raw predictions from matcher
            requirements_df: Optional requirements context
            
        Returns:
            Dict with enhanced predictions, evaluation results, and dashboard path
        """
        
        logger.info("🚀 Starting enhanced evaluation...")
        
        # Step 1: Always add quality analysis
        enhanced_df = self._add_quality_analysis(predictions_df, requirements_df)
        
        # Step 2: Compute evaluation metrics if ground truth available
        evaluation_results = {}
        if self.has_ground_truth:
            evaluation_results = self._compute_evaluation_metrics(enhanced_df)
        
        # Step 3: Always create dashboard (adapts to available data)
        dashboard_path = self._create_dashboard(enhanced_df, evaluation_results, requirements_df)
        self.quality_analyzer.create_excel_report(enhanced_df, repo_manager=self.repo_manager)
        # Step 4: Print summary
        self._print_evaluation_summary(enhanced_df, evaluation_results)
        
        return {
            'enhanced_predictions': enhanced_df,
            'evaluation_results': evaluation_results,
            'dashboard_path': dashboard_path,
            'has_ground_truth': self.has_ground_truth,
            'ground_truth': self.ground_truth
        }
    
    def _print_evaluation_summary(self, enhanced_df: pd.DataFrame, evaluation_results: Dict):
        """Print comprehensive evaluation summary."""
        
        print("\n" + "=" * 70)
        print("ENHANCED EVALUATION SUMMARY")
        print("=" * 70)
        
        print(f"📊 Dataset Overview:")
        print(f"  Total predictions: {len(enhanced_df)}")
        print(f"  Unique requirements: {enhanced_df['ID'].nunique() if 'ID' in enhanced_df.columns else 'N/A'}")
        print(f"  Dashboard: {evaluation_results.get('dashboard_path', 'N/A')}")
        
        # Quality summary
        if 'Quality_Grade' in enhanced_df.columns:
            quality_dist = enhanced_df['Quality_Grade'].value_counts()
            avg_quality = enhanced_df['Quality_Score'].mean()
            needs_improvement = len(enhanced_df[enhanced_df['Quality_Grade'].isin(['POOR', 'CRITICAL'])])
            
            print(f"\n🎯 Quality Analysis:")
            print(f"  Average quality score: {avg_quality:.1f}/100")
            print(f"  Requirements needing improvement: {needs_improvement}")
            print(f"  Quality distribution:")
            for grade, count in quality_dist.items():
                percentage = count / len(enhanced_df) * 100
                print(f"    {grade}: {count} ({percentage:.1f}%)")
        
        # Evaluation metrics summary
        if evaluation_results and 'aggregate_metrics' in evaluation_results:
            metrics = evaluation_results['aggregate_metrics']
            coverage = evaluation_results.get('coverage', 0)
            
            print(f"\n📈 Evaluation Metrics:")
            print(f"  Coverage: {coverage:.1%}")
            if 'MRR' in metrics:
                print(f"  MRR: {metrics['MRR']['mean']:.3f}")
            if 'f1_at_5' in metrics:
                print(f"  F1@5: {metrics['f1_at_5']['mean']:.3f}")
        
        # Discovery summary
        if evaluation_results and 'discovery_analysis' in evaluation_results:
            discovery = evaluation_results['discovery_analysis']['summary']
            print(f"\n🔍 Discovery Analysis:")
            print(f"  High-scoring discoveries: {discovery.get('total_high_scoring_misses', 0)}")
            print(f"  Discovery rate: {discovery.get('discovery_rate', 0):.1%}")
        
        print(f"\n✅ Analysis complete! Check dashboard for interactive exploration.")


def main():
    """Example usage of enhanced evaluator."""
    
    # Example with ground truth
    evaluator_with_gt = EnhancedMatchingEvaluator("manual_matches.csv")
    
    # Example without ground truth  
    evaluator_exploration = EnhancedMatchingEvaluator(None)
    
    # Load sample predictions (this would come from your matcher)
    try:
        predictions_df = pd.read_csv("enhanced_workflow_matches.csv")
        requirements_df = pd.read_csv("requirements.csv")
        
        # Run evaluation
        results = evaluator_with_gt.evaluate_predictions(predictions_df, requirements_df)
        
        print(f"✅ Evaluation complete!")
        print(f"📊 Dashboard: {results['dashboard_path']}")
        print(f"🎯 Has ground truth: {results['has_ground_truth']}")
        
    except FileNotFoundError as e:
        print(f"Example files not found: {e}")


if __name__ == "__main__":
    main()