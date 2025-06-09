"""
Data Processing - Clean, normalize, and structure data for dashboard consumption
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import re
from datetime import datetime

class DataProcessor:
    """Handles all data processing, cleaning, and structuring."""
    
    def __init__(self, ground_truth: Optional[Dict] = None):
        self.ground_truth = ground_truth or {}
    
    def process_evaluation_data(self, evaluation_results: Dict, 
                              predictions_df: pd.DataFrame,
                              requirements_df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Main data processing pipeline."""
        exploration_coverage_rate = 0.0
        if not evaluation_results.get('coverage'):
            # Calculate what percentage of unique requirements have at least one match
            if 'ID' in predictions_df.columns:
                # Initialize total_unique_reqs first
                total_unique_reqs = predictions_df['ID'].nunique()
                
                # Use requirements_df count if available and non-empty (more accurate)
                if requirements_df is not None and len(requirements_df) > 0:
                    total_unique_reqs = len(requirements_df)
                
                # All requirements in predictions_df have at least one match by definition
                covered_reqs = predictions_df['ID'].nunique()
                exploration_coverage_rate = covered_reqs / total_unique_reqs if total_unique_reqs > 0 else 0.0

        processed = {
            'metadata': self._extract_metadata(evaluation_results),
            'performance_metrics': self._process_performance_metrics(evaluation_results),
            'discovery_data': self._process_discovery_data(evaluation_results),
            'predictions_data': self._process_predictions_data(predictions_df, requirements_df),
            'score_distributions': self._process_score_distributions(evaluation_results),
            'summary_stats': self._calculate_summary_stats(evaluation_results, predictions_df),
            'quality_data': self._extract_quality_data(predictions_df),
            'coverage_data': self._generate_coverage_analysis(predictions_df, requirements_df)
        }
        if processed['discovery_data'].get('coverage_rate', 0) == 0 and exploration_coverage_rate > 0:
            processed['discovery_data']['coverage_rate'] = exploration_coverage_rate

        return processed
    
    def _extract_metadata(self, results: Dict) -> Dict[str, Any]:
        """Extract metadata about the evaluation."""
        
        # Use corrected total_requirements count
        total_requirements = results.get('total_requirements', 0)
        
        # If still 0, try to get from quality metadata or other sources
        if total_requirements == 0:
            quality_meta = results.get('quality_metadata', {})
            total_requirements = quality_meta.get('unique_requirements_analyzed', 0)
            
            # Final fallback: count unique from predictions
            if total_requirements == 0 and 'predictions_df' in results:
                predictions_df = results['predictions_df']
                if hasattr(predictions_df, 'nunique'):
                    id_col = 'ID' if 'ID' in predictions_df.columns else 'requirement_id'
                    if id_col in predictions_df.columns:
                        total_requirements = predictions_df[id_col].nunique()
        
        return {
            'generated_at': datetime.now().isoformat(),
            'total_requirements': total_requirements,
            'total_predictions': len(results.get('predictions_df', [])),
            'coverage': results.get('coverage', 0),
            'tool_version': "modular_dashboard_v1.0"
        }

    
    def _process_performance_metrics(self, results: Dict) -> Dict[str, Any]:
        """Extract and structure performance metrics."""
        if 'aggregate_metrics' not in results:
            return {}
        
        agg = results['aggregate_metrics']
        metrics = {}
        
        # Extract k-based metrics
        for k in [1, 3, 5, 10]:
            for metric in ['precision', 'recall', 'f1', 'ndcg']:
                key = f'{metric}_at_{k}'
                if key in agg:
                    metrics[key] = {
                        'mean': agg[key]['mean'],
                        'std': agg[key]['std'],
                        'k': k,
                        'metric_type': metric
                    }
        
        # Add MRR
        if 'MRR' in agg:
            metrics['MRR'] = {
                'mean': agg['MRR']['mean'],
                'std': agg['MRR']['std']
            }
        
        return metrics
    


    def _process_discovery_data(self, results: Dict) -> Dict[str, Any]:
        """Process discovery analysis data."""
        if 'discovery_analysis' not in results:
            return {}
        
        discovery = results['discovery_analysis']
        
        # Calculate coverage rate from evaluation results when available
        coverage_rate = 0.0
        if 'coverage' in results:
            coverage_rate = results['coverage']
        elif 'total_requirements' in results and 'covered_requirements' in results:
            total_reqs = results['total_requirements']
            covered_reqs = results['covered_requirements'] 
            coverage_rate = covered_reqs / total_reqs if total_reqs > 0 else 0.0
        
        # Get ground truth for manual matches lookup
        ground_truth = results.get('ground_truth', {})
        
        # Enhance high_scoring_misses with manual match info
        enhanced_misses = []
        for miss in discovery.get('high_scoring_misses', []):
            enhanced_miss = miss.copy()
            
            # Add manual matches for this requirement
            req_id = miss['requirement_id']
            manual_matches = []
            if req_id in ground_truth:
                manual_matches = [item['original_activity'] for item in ground_truth[req_id]]
            
            enhanced_miss['manual_matches'] = manual_matches
            enhanced_misses.append(enhanced_miss)
        
        # Enhance score_gaps with manual match info
        enhanced_gaps = []
        for gap in discovery.get('score_gaps', []):
            enhanced_gap = gap.copy()
            
            # Add manual matches for this requirement
            req_id = gap['requirement_id']
            manual_matches = []
            if req_id in ground_truth:
                manual_matches = [item['original_activity'] for item in ground_truth[req_id]]
            
            enhanced_gap['manual_matches'] = manual_matches
            enhanced_gaps.append(enhanced_gap)
        
        return {
            'high_scoring_misses': enhanced_misses,
            'score_gaps': enhanced_gaps,
            'summary': discovery.get('summary', {}),
            'coverage_rate': coverage_rate,  # NOW coverage_rate is defined
            'top_examples': sorted(enhanced_misses, key=lambda x: x['score'], reverse=True)[:20]
        }

    
    def _process_predictions_data(self, predictions_df: pd.DataFrame, 
                                requirements_df: Optional[pd.DataFrame] = None) -> List[Dict]:
        """Process and enrich predictions data - WITH DEBUGGING."""
        
        print(f"\n🔍 PREDICTIONS DATA PROCESSOR DEBUG:")
        print(f"  - Input predictions_df shape: {predictions_df.shape}")
        print(f"  - Input predictions_df columns: {list(predictions_df.columns)}")
        print(f"  - First 3 rows preview:")
        for i, (_, row) in enumerate(predictions_df.head(3).iterrows()):
            print(f"    Row {i}: {dict(row)}")
        
        # Standardize column names
        id_col = 'ID' if 'ID' in predictions_df.columns else 'requirement_id'
        activity_col = 'Activity Name' if 'Activity Name' in predictions_df.columns else 'activity_name'
        score_col = 'Combined Score' if 'Combined Score' in predictions_df.columns else 'score'
        
        print(f"  - Using columns: id='{id_col}', activity='{activity_col}', score='{score_col}'")
        
        # NEW: Create requirement name lookup from requirements_df if available
        req_name_lookup = {}
        req_text_lookup = {}
        if requirements_df is not None:
            print(f"  - Requirements_df shape: {requirements_df.shape}")
            print(f"  - Requirements_df columns: {list(requirements_df.columns)}")
            
            req_id_col = 'ID' if 'ID' in requirements_df.columns else 'requirement_id'
            req_name_col = None
            req_text_col = None
            
            # Find requirement name column
            for col in ['Requirement Name', 'Name', 'Requirement_Name', 'requirement_name']:
                if col in requirements_df.columns:
                    req_name_col = col
                    print(f"  - Found requirement name column: {col}")
                    break
            
            # Find requirement text column  
            for col in ['Requirement Text', 'Text', 'Requirement_Text', 'requirement_text', 'Description']:
                if col in requirements_df.columns:
                    req_text_col = col
                    print(f"  - Found requirement text column: {col}")
                    break
            
            if req_name_col:
                req_name_lookup = dict(zip(requirements_df[req_id_col].astype(str), 
                                        requirements_df[req_name_col].fillna('N/A')))
                print(f"  - Created name lookup with {len(req_name_lookup)} entries")
            
            if req_text_col:
                req_text_lookup = dict(zip(requirements_df[req_id_col].astype(str), 
                                        requirements_df[req_text_col].fillna('N/A')))
                print(f"  - Created text lookup with {len(req_text_lookup)} entries")
        
        processed_predictions = []
        
        print(f"  - Processing {len(predictions_df)} prediction rows...")
        
        for idx, (_, row) in enumerate(predictions_df.iterrows()):
            req_id = str(row[id_col])
            
            # Basic data
            pred_data = {
                'requirement_id': req_id,
                'activity_name': str(row[activity_col]),
                'combined_score': row.get(score_col, 0),
                # NEW: Try predictions_df first, then lookup, then fallback
                'requirement_name': (row.get('Requirement Name') or 
                                req_name_lookup.get(req_id) or 
                                f"Requirement {req_id}"),
                'requirement_text': (row.get('Requirement Text') or 
                                req_text_lookup.get(req_id) or 
                                'Requirement text not available')
            }
                
            # Component scores
            pred_data.update({
                'semantic_score': row.get('Dense Semantic', row.get('Semantic Score', 0)),
                'bm25_score': row.get('BM25 Score', 0),
                'syntactic_score': row.get('Syntactic Score', 0),
                'domain_score': row.get('Domain Weighted', row.get('Domain Score', 0)),
                'query_expansion_score': row.get('Query Expansion', 0)
            })
            
            # Classify score level
            score = pred_data['combined_score']
            if score >= 1.0:
                pred_data['score_class'] = 'high'
            elif score >= 0.6:
                pred_data['score_class'] = 'medium'
            else:
                pred_data['score_class'] = 'low'
            
            processed_predictions.append(pred_data)
            
            # Debug first few predictions
            if idx < 3:
                print(f"    Processed prediction {idx}: {req_id} -> {pred_data['activity_name'][:50]}...")
        
        print(f"✅ PREDICTIONS PROCESSOR RESULT: {len(processed_predictions)} predictions processed")
        print(f"  - Sample processed data:")
        if processed_predictions:
            sample = processed_predictions[0]
            print(f"    {sample['requirement_id']}: {sample['activity_name'][:30]}... (score: {sample['combined_score']})")
        
        return processed_predictions    
    def _process_score_distributions(self, results: Dict) -> Dict[str, Any]:
        """Process score distribution data."""
        return results.get('score_distributions', {})
    
    def _calculate_summary_stats(self, results: Dict, predictions_df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate dashboard summary statistics with corrected counts."""
        
        # Use corrected total_requirements
        total_requirements = results.get('total_requirements', 0)
        if total_requirements == 0:
            # Fallback: count unique requirements from predictions
            id_col = 'ID' if 'ID' in predictions_df.columns else 'requirement_id'
            if id_col in predictions_df.columns:
                total_requirements = predictions_df[id_col].nunique()
        
        stats = {
            'total_requirements': total_requirements,
            'total_predictions': len(predictions_df),
            'coverage_percentage': results.get('coverage', 0) * 100,
            'f1_at_5': 0,
            'discoveries_count': 0,
            'score_gaps_count': 0
        }
        
        # F1@5 score
        if 'aggregate_metrics' in results:
            stats['f1_at_5'] = results['aggregate_metrics'].get('f1_at_5', {}).get('mean', 0)
        
        # Discovery stats
        if 'discovery_analysis' in results:
            discovery = results['discovery_analysis']['summary']
            stats['discoveries_count'] = discovery.get('total_high_scoring_misses', 0)
            stats['score_gaps_count'] = discovery.get('score_gaps_count', 0)
        
        # Quality stats - use requirement-level counts if available
        quality_meta = results.get('quality_metadata', {})
        if quality_meta:
            stats['unique_requirements_analyzed'] = quality_meta.get('unique_requirements_analyzed', 0)
            stats['quality_distribution'] = quality_meta.get('quality_distribution_by_requirement', {})
        
        return stats
    
    def _extract_quality_data(self, predictions_df: pd.DataFrame) -> Dict[str, Any]:
        """Extract quality analysis data from predictions DataFrame."""
        quality_data = {
            'has_quality_data': False,
            'quality_distribution': {},
            'quality_score_correlation': {},
            'quality_recommendations': [],
            'quality_stats': {},
            'unique_requirements_quality': {}  # NEW: requirement-level quality
        }
        
        # Check for quality columns
        quality_columns = {
            'Quality_Grade': 'Quality Grade',
            'Quality_Score': 'Quality Score', 
            'Clarity_Score': 'Clarity Score',
            'Completeness_Score': 'Completeness Score',
            'Verifiability_Score': 'Verifiability Score',
            'Atomicity_Score': 'Atomicity Score',
            'Consistency_Score': 'Consistency Score'
        }
        
        available_quality_cols = {}
        for col_var, col_display in quality_columns.items():
            if col_var in predictions_df.columns:
                available_quality_cols[col_var] = col_display
            elif col_display in predictions_df.columns:
                available_quality_cols[col_display] = col_display
        
        if not available_quality_cols:
            return quality_data
        
        quality_data['has_quality_data'] = True
        
        # Get unique requirements quality data (not by match)
        id_col = 'ID' if 'ID' in predictions_df.columns else 'requirement_id'
        if id_col in predictions_df.columns:
            # Group by requirement and get first quality values (they should be same for same req)
            unique_req_quality = predictions_df.groupby(id_col).first()
            
            # Quality grade distribution BY REQUIREMENT
            grade_col = None
            for col in ['Quality_Grade', 'Quality Grade']:
                if col in predictions_df.columns:
                    grade_col = col
                    break
            
            if grade_col:
                # Count unique requirements by grade, not matches
                grade_dist = unique_req_quality[grade_col].value_counts().to_dict()
                quality_data['quality_distribution'] = grade_dist
                quality_data['unique_requirements_quality']['grade_distribution'] = grade_dist
                quality_data['unique_requirements_quality']['total_requirements'] = len(unique_req_quality)
            
            # Quality score statistics BY REQUIREMENT
            score_col = None
            for col in ['Quality_Score', 'Quality Score']:
                if col in predictions_df.columns:
                    score_col = col
                    break
            
            if score_col:
                req_scores = unique_req_quality[score_col]
                quality_data['quality_stats'] = {
                    'mean': req_scores.mean(),
                    'median': req_scores.median(),
                    'std': req_scores.std(),
                    'min': req_scores.min(),
                    'max': req_scores.max(),
                    'count': len(req_scores)  # Number of requirements, not matches
                }
                quality_data['unique_requirements_quality']['score_stats'] = quality_data['quality_stats']
        else:
            # Fallback to match-level analysis if no ID column
            grade_col = None
            for col in ['Quality_Grade', 'Quality Grade']:
                if col in predictions_df.columns:
                    grade_col = col
                    break
            
            if grade_col:
                grade_dist = predictions_df[grade_col].value_counts().to_dict()
                quality_data['quality_distribution'] = grade_dist
        
        # Quality-Score correlation analysis
        combined_score_col = None
        for col in ['Combined Score', 'Combined_Score', 'score']:
            if col in predictions_df.columns:
                combined_score_col = col
                break
        
        if score_col and combined_score_col:
            # Use requirement-level data if available
            if id_col in predictions_df.columns:
                analysis_df = predictions_df.groupby(id_col).agg({
                    score_col: 'first',  # Quality is same per requirement
                    combined_score_col: 'max'  # Best match score per requirement
                }).reset_index()
            else:
                analysis_df = predictions_df
            
            # Analyze correlation between quality and match scores
            high_quality = analysis_df[analysis_df[score_col] >= 70]
            medium_quality = analysis_df[(analysis_df[score_col] >= 50) & (analysis_df[score_col] < 70)]
            low_quality = analysis_df[analysis_df[score_col] < 50]
            
            quality_data['quality_score_correlation'] = {
                'high_quality_matches': {
                    'count': len(high_quality),
                    'avg_match_score': high_quality[combined_score_col].mean() if len(high_quality) > 0 else 0,
                    'good_matches': len(high_quality[high_quality[combined_score_col] >= 0.6]) if len(high_quality) > 0 else 0
                },
                'medium_quality_matches': {
                    'count': len(medium_quality),
                    'avg_match_score': medium_quality[combined_score_col].mean() if len(medium_quality) > 0 else 0,
                    'good_matches': len(medium_quality[medium_quality[combined_score_col] >= 0.6]) if len(medium_quality) > 0 else 0
                },
                'low_quality_matches': {
                    'count': len(low_quality),
                    'avg_match_score': low_quality[combined_score_col].mean() if len(low_quality) > 0 else 0,
                    'good_matches': len(low_quality[low_quality[combined_score_col] >= 0.6]) if len(low_quality) > 0 else 0
                }
            }
        
        # Generate recommendations
        recommendations = []
        if grade_col and 'quality_distribution' in quality_data:
            grade_dist = quality_data['quality_distribution']
            poor_count = grade_dist.get('POOR', 0) + grade_dist.get('CRITICAL', 0)
            if poor_count > 0:
                recommendations.append({
                    'priority': 'high',
                    'text': f'Review and improve {poor_count} requirements with POOR/CRITICAL quality grades'
                })
            
            fair_count = grade_dist.get('FAIR', 0)
            if fair_count > 0:
                recommendations.append({
                    'priority': 'medium', 
                    'text': f'Consider enhancing {fair_count} requirements with FAIR quality grade'
                })
        
        if score_col and 'quality_stats' in quality_data:
            avg_score = quality_data['quality_stats']['mean']
            if avg_score < 60:
                recommendations.append({
                    'priority': 'high',
                    'text': f'Overall quality score ({avg_score:.1f}) is below recommended threshold (60)'
                })
        
        quality_data['quality_recommendations'] = recommendations
        
        return quality_data
    
    def _generate_coverage_analysis(self, predictions_df: pd.DataFrame, 
                                  requirements_df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """Generate coverage analysis data."""
        coverage_data = {
            'has_coverage_data': True,
            'score_distribution': {},
            'confidence_levels': {},
            'review_priorities': {},
            'coverage_stats': {},
            'effort_estimation': {}
        }
        
        # Score distribution analysis
        combined_score_col = None
        for col in ['Combined Score', 'Combined_Score', 'score']:
            if col in predictions_df.columns:
                combined_score_col = col
                break
        
        if combined_score_col:
            scores = predictions_df[combined_score_col]
            
            # Score ranges
            high_scores = len(scores[scores >= 1.0])
            medium_scores = len(scores[(scores >= 0.6) & (scores < 1.0)]) 
            low_scores = len(scores[scores < 0.6])
            
            coverage_data['score_distribution'] = {
                'high': {'count': high_scores, 'percentage': high_scores / len(scores) * 100},
                'medium': {'count': medium_scores, 'percentage': medium_scores / len(scores) * 100},
                'low': {'count': low_scores, 'percentage': low_scores / len(scores) * 100}
            }
            
            coverage_data['coverage_stats'] = {
                'total_matches': len(predictions_df),
                'avg_score': scores.mean(),
                'median_score': scores.median(),
                'std_score': scores.std(),
                'min_score': scores.min(),
                'max_score': scores.max()
            }
        
        # Requirements coverage analysis
        if 'ID' in predictions_df.columns or 'requirement_id' in predictions_df.columns:
            id_col = 'ID' if 'ID' in predictions_df.columns else 'requirement_id'
            unique_reqs = predictions_df[id_col].nunique()
            
            # Requirements by score level
            req_scores = predictions_df.groupby(id_col)[combined_score_col].max() if combined_score_col else pd.Series()
            
            if not req_scores.empty:
                reqs_high = len(req_scores[req_scores >= 1.0])
                reqs_medium = len(req_scores[(req_scores >= 0.6) & (req_scores < 1.0)])
                reqs_low = len(req_scores[req_scores < 0.6])
                
                coverage_data['requirements_coverage'] = {
                    'total_requirements': unique_reqs,
                    'high_confidence': {'count': reqs_high, 'percentage': reqs_high / unique_reqs * 100},
                    'medium_confidence': {'count': reqs_medium, 'percentage': reqs_medium / unique_reqs * 100},
                    'low_confidence': {'count': reqs_low, 'percentage': reqs_low / unique_reqs * 100}
                }
        
        # Review effort estimation
        if combined_score_col:
            auto_approve = len(predictions_df[predictions_df[combined_score_col] >= 1.0])
            quick_review = len(predictions_df[(predictions_df[combined_score_col] >= 0.7) & (predictions_df[combined_score_col] < 1.0)])
            detailed_review = len(predictions_df[(predictions_df[combined_score_col] >= 0.4) & (predictions_df[combined_score_col] < 0.7)])
            manual_analysis = len(predictions_df[predictions_df[combined_score_col] < 0.4])
            
            coverage_data['effort_estimation'] = {
                'auto_approve': {'count': auto_approve, 'hours': auto_approve * 0.25},
                'quick_review': {'count': quick_review, 'hours': quick_review * 0.5},
                'detailed_review': {'count': detailed_review, 'hours': detailed_review * 2.0},
                'manual_analysis': {'count': manual_analysis, 'hours': manual_analysis * 4.0}
            }
            
            total_hours = (auto_approve * 0.25 + quick_review * 0.5 + 
                         detailed_review * 2.0 + manual_analysis * 4.0)
            coverage_data['effort_estimation']['total_hours'] = total_hours
        
        return coverage_data