#!/usr/bin/env python3
"""
Fixed Simple Requirements Matching Evaluation
============================================

Enhanced version with comprehensive debugging and fixes for:
1. Column name mismatches (Combined_Score vs Combined Score)
2. Ground truth parsing issues
3. Requirement ID consistency problems
4. Detailed debug output to identify issues
5. Proper repository structure and path resolution
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path
import json
from typing import Dict, List, Tuple
from collections import defaultdict
import re

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import repository utilities
try:
    from src.utils.repository_setup import RepositoryStructureManager
    from src.utils.path_resolver import SmartPathResolver
    UTILS_AVAILABLE = True
except ImportError:
    print("⚠️ Repository utils not available - using basic file handling")
    UTILS_AVAILABLE = False

class FixedSimpleEvaluator:
    """Enhanced evaluator with comprehensive debugging and fixes."""
    
    def __init__(self):
        self.results = {}
        self.debug_mode = True
        
        # Setup paths properly
        if UTILS_AVAILABLE:
            self.repo_manager = RepositoryStructureManager("outputs")
            self.path_resolver = SmartPathResolver(self.repo_manager)
        else:
            self.repo_manager = None
            self.path_resolver = None
    
    def get_file_paths(self, matches_file: str, ground_truth_file: str, requirements_file: str) -> Dict[str, str]:
        """Get proper file paths using repository utilities."""
        
        if UTILS_AVAILABLE:
            # Define file mapping for input files
            file_mapping = {
                'ground_truth': ground_truth_file,
                'requirements': requirements_file
            }
            
            # Resolve input file paths
            resolved_paths = self.path_resolver.resolve_input_files(file_mapping)
            
            # Add matcher results path (always in outputs)
            resolved_paths['matches'] = matches_file
            
            return resolved_paths
        else:
            # Fallback to manual paths
            return {
                'matches': matches_file,
                'ground_truth': f"data/raw/{ground_truth_file}",
                'requirements': f"data/raw/{requirements_file}"
            }
    
    def evaluate_matches(self, 
                        matches_file: str = "outputs/matching_results/aerospace_matches.csv",
                        ground_truth_file: str = "manual_matches.csv",
                        requirements_file: str = "requirements.csv") -> Dict:
        """Run enhanced evaluation with debugging."""
        
        print("🔍 FIXED SIMPLE MATCHING EVALUATION")
        print("=" * 50)
        print("🐛 Debug mode: ON - Will show detailed diagnostics")
        
        # Get proper file paths
        file_paths = self.get_file_paths(matches_file, ground_truth_file, requirements_file)
        
        # 1. Load data with enhanced error handling
        print("\n📂 Loading data...")
        matches_df = self._load_matches_enhanced(file_paths['matches'])
        ground_truth = self._load_ground_truth_enhanced(file_paths['ground_truth'])
        requirements_df = self._load_requirements_enhanced(file_paths['requirements'])
        
        if matches_df is None or ground_truth is None:
            return {"error": "Could not load required files"}
        
        # 2. Data quality checks
        print("\n🔍 DATA QUALITY CHECKS")
        self._debug_data_quality(matches_df, ground_truth)
        
        # 3. Column analysis
        print("\n📊 COLUMN ANALYSIS")
        score_col = self._identify_score_column(matches_df)
        req_col, act_col = self._identify_key_columns(matches_df)
        
        # 4. Compute metrics with debugging
        print("\n📊 Computing evaluation metrics...")
        metrics = self._compute_metrics_enhanced(matches_df, ground_truth, score_col, req_col, act_col)
        
        # 5. Deep analysis
        print("\n🔬 Analyzing results...")
        analysis = self._analyze_results_enhanced(matches_df, ground_truth, requirements_df, score_col, req_col, act_col)
        
        # 6. Generate report
        print("\n📝 Generating report...")
        report = self._generate_report_enhanced(metrics, analysis)
        
        self.results = {
            'metrics': metrics,
            'analysis': analysis,
            'report': report,
            'debugging_info': {
                'score_column': score_col,
                'req_column': req_col,
                'activity_column': act_col,
                'file_paths': file_paths
            }
        }
        
        # 7. Save results
        self._save_results_enhanced()
        
        return self.results
    
    def _load_matches_enhanced(self, matches_file: str) -> pd.DataFrame:
        """Enhanced matcher results loading with debugging."""
        try:
            # Try different encodings
            encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252']
            df = None
            
            for encoding in encodings:
                try:
                    df = pd.read_csv(matches_file, encoding=encoding)
                    print(f"✅ Loaded {len(df)} matches from {Path(matches_file).name} (encoding: {encoding})")
                    break
                except UnicodeDecodeError:
                    continue
            
            if df is None:
                raise Exception("Could not read file with any encoding")
            
            print(f"   📋 Columns: {list(df.columns)}")
            print(f"   📏 Shape: {df.shape}")
            
            # Check for common issues
            if len(df) == 0:
                print("   ⚠️  WARNING: No data rows found!")
            
            return df
            
        except Exception as e:
            print(f"❌ Could not load matches: {e}")
            return None
    
    def _load_ground_truth_enhanced(self, gt_file: str) -> Dict:
        """Enhanced ground truth loading with debugging."""
        try:
            # Try different encodings
            encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252']
            df = None
            
            for encoding in encodings:
                try:
                    df = pd.read_csv(gt_file, encoding=encoding)
                    print(f"✅ Loaded ground truth from {Path(gt_file).name} (encoding: {encoding})")
                    break
                except UnicodeDecodeError:
                    continue
            
            if df is None:
                raise Exception("Could not read file with any encoding")
            
            print(f"   📋 Columns: {list(df.columns)}")
            print(f"   📏 Shape: {df.shape}")
            
            # Enhanced normalization function
            def normalize_text(text):
                if pd.isna(text) or text == '':
                    return ""
                text = str(text).strip()
                # Remove leading numbers (e.g., "1.2.3 Activity Name" -> "Activity Name")
                text = re.sub(r'^\d+(\.\d+)*\s+', '', text)
                # Remove context information in parentheses
                text = text.split('(context')[0]
                # Clean up spacing and normalize
                text = re.sub(r'\s+', ' ', text)  # Multiple spaces to single
                return text.strip().lower().replace("-", " ")
            
            ground_truth = defaultdict(list)
            
            # Find columns (handle different naming)
            req_col = None
            act_col = None
            
            for col in df.columns:
                if col.lower() in ['id', 'requirement_id', 'req_id']:
                    req_col = col
                elif col.lower() in ['satisfied by', 'satisfied_by', 'activity_name', 'activities']:
                    act_col = col
            
            if not req_col or not act_col:
                print(f"   ⚠️  Column detection: req_col={req_col}, act_col={act_col}")
                # Fallback to first two columns
                req_col = df.columns[0]
                act_col = df.columns[-1] if len(df.columns) > 1 else df.columns[0]
                print(f"   🔄 Using fallback: req_col={req_col}, act_col={act_col}")
            
            processed_count = 0
            
            for _, row in df.iterrows():
                req_id = str(row[req_col]).strip()
                satisfied_by = str(row.get(act_col, '')).strip()
                
                if satisfied_by and satisfied_by.lower() != 'nan':
                    # Split multiple activities (handle comma-separated)
                    activities = [activity.strip() for activity in satisfied_by.split(',')]
                    
                    for activity in activities:
                        if activity:
                            normalized_activity = normalize_text(activity)
                            if normalized_activity:
                                ground_truth[req_id].append({
                                    'normalized': normalized_activity,
                                    'original': activity.strip()
                                })
                                processed_count += 1
            
            print(f"   📊 Ground truth for {len(ground_truth)} requirements")
            print(f"   📊 Total {processed_count} activity mappings")
            print(f"   📋 Using columns: '{req_col}' → '{act_col}'")
            
            return dict(ground_truth)
            
        except Exception as e:
            print(f"❌ Could not load ground truth: {e}")
            return None
    
    def _load_requirements_enhanced(self, req_file: str) -> pd.DataFrame:
        """Enhanced requirements loading."""
        try:
            encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252']
            
            for encoding in encodings:
                try:
                    df = pd.read_csv(req_file, encoding=encoding)
                    print(f"✅ Loaded {len(df)} requirements for context (encoding: {encoding})")
                    return df
                except UnicodeDecodeError:
                    continue
            
            raise Exception("Could not read with any encoding")
            
        except Exception as e:
            print(f"⚠️ Could not load requirements (optional): {e}")
            return None
    
    def _debug_data_quality(self, matches_df: pd.DataFrame, ground_truth: Dict):
        """Debug data quality issues."""
        print("   🔍 Checking data consistency...")
        
        # Check matches data
        req_ids_matches = set(matches_df.iloc[:, 0].astype(str).str.strip())
        req_ids_gt = set(ground_truth.keys())
        
        print(f"   📊 Requirement IDs in matches: {len(req_ids_matches)}")
        print(f"   📊 Requirement IDs in ground truth: {len(req_ids_gt)}")
        
        # Check overlap
        overlap = req_ids_matches & req_ids_gt
        only_matches = req_ids_matches - req_ids_gt
        only_gt = req_ids_gt - req_ids_matches
        
        print(f"   🎯 Overlapping IDs: {len(overlap)}")
        if only_matches:
            print(f"   ⚠️  Only in matches: {len(only_matches)} (e.g., {list(only_matches)[:3]})")
        if only_gt:
            print(f"   ⚠️  Only in ground truth: {len(only_gt)} (e.g., {list(only_gt)[:3]})")
        
        # Check for duplicates
        match_counts = matches_df.iloc[:, 0].value_counts()
        duplicates = match_counts[match_counts > 5]  # More than 5 matches per requirement
        if len(duplicates) > 0:
            print(f"   📈 Requirements with many matches: {len(duplicates)}")
    
    def _identify_score_column(self, matches_df: pd.DataFrame) -> str:
        """Identify the correct score column."""
        possible_score_cols = [
            'Combined_Score', 'Combined Score', 'combined_score', 'score', 
            'Score', 'total_score', 'Total_Score', 'final_score'
        ]
        
        score_col = None
        for col in possible_score_cols:
            if col in matches_df.columns:
                score_col = col
                break
        
        if not score_col:
            # Look for any column with 'score' in the name
            score_cols = [col for col in matches_df.columns if 'score' in col.lower()]
            if score_cols:
                score_col = score_cols[0]
            else:
                # Use last numeric column as fallback
                numeric_cols = matches_df.select_dtypes(include=[np.number]).columns
                score_col = numeric_cols[-1] if len(numeric_cols) > 0 else None
        
        print(f"   🎯 Score column identified: '{score_col}'")
        
        if score_col:
            sample_scores = matches_df[score_col].head()
            print(f"   📊 Sample scores: {sample_scores.tolist()}")
            print(f"   📊 Score range: {matches_df[score_col].min():.3f} - {matches_df[score_col].max():.3f}")
        
        return score_col
    
    def _identify_key_columns(self, matches_df: pd.DataFrame) -> Tuple[str, str]:
        """Identify requirement and activity columns."""
        req_col = None
        act_col = None
        
        # Find requirement column
        req_candidates = ['Requirement_ID', 'ID', 'req_id', 'requirement_id']
        for col in req_candidates:
            if col in matches_df.columns:
                req_col = col
                break
        
        if not req_col:
            req_col = matches_df.columns[0]  # Fallback to first column
        
        # Find activity column  
        act_candidates = ['Activity_Name', 'Activity Name', 'activity_name', 'Activity']
        for col in act_candidates:
            if col in matches_df.columns:
                act_col = col
                break
        
        if not act_col:
            # Look for column with 'activity' in name
            act_cols = [col for col in matches_df.columns if 'activity' in col.lower()]
            act_col = act_cols[0] if act_cols else matches_df.columns[1]
        
        print(f"   🎯 Key columns: req='{req_col}', activity='{act_col}'")
        
        return req_col, act_col
    
    def _normalize_activity_enhanced(self, activity: str) -> str:
        """Enhanced activity normalization matching ground truth processing."""
        if pd.isna(activity) or activity == '':
            return ""
        
        text = str(activity).strip()
        # Remove leading numbers
        text = re.sub(r'^\d+(\.\d+)*\s+', '', text)
        # Remove context information
        text = text.split('(context')[0]
        # Clean up spacing
        text = re.sub(r'\s+', ' ', text)
        return text.strip().lower().replace("-", " ")
    
    def _compute_metrics_enhanced(self, matches_df: pd.DataFrame, ground_truth: Dict, 
                                 score_col: str, req_col: str, act_col: str) -> Dict:
        """Enhanced metrics computation with detailed debugging."""
        
        # Group matches by requirement
        matches_by_req = defaultdict(list)
        
        for _, row in matches_df.iterrows():
            req_id = str(row[req_col]).strip()
            activity = self._normalize_activity_enhanced(row[act_col])
            score = row.get(score_col, 0) if score_col else 0
            
            matches_by_req[req_id].append({
                'activity': activity,
                'original': row[act_col],
                'score': float(score) if pd.notna(score) else 0.0
            })
        
        # Sort matches by score for each requirement
        for req_id in matches_by_req:
            matches_by_req[req_id].sort(key=lambda x: x['score'], reverse=True)
        
        # ENHANCED DEBUG SECTION
        print(f"   🔍 DETAILED DEBUG ANALYSIS:")
        
        # Sample requirement deep dive
        common_reqs = set(matches_by_req.keys()) & set(ground_truth.keys())
        if common_reqs:
            sample_req = list(common_reqs)[0]
            print(f"\n   📋 Deep dive for requirement: {sample_req}")
            
            # Ground truth
            gt_items = ground_truth[sample_req]
            gt_normalized = [item['normalized'] for item in gt_items]
            gt_original = [item['original'] for item in gt_items]
            
            print(f"   📋 Ground truth ({len(gt_normalized)} items):")
            for i, (norm, orig) in enumerate(zip(gt_normalized, gt_original)):
                print(f"      {i+1}. '{norm}' (from: '{orig}')")
            
            # Matcher predictions
            if sample_req in matches_by_req:
                pred_items = matches_by_req[sample_req][:5]
                print(f"   🎯 Top 5 predictions:")
                for i, pred in enumerate(pred_items):
                    match_status = "✅ MATCH" if pred['activity'] in gt_normalized else "❌ MISS"
                    print(f"      {i+1}. '{pred['activity']}' (score: {pred['score']:.3f}) {match_status}")
                    print(f"         Original: '{pred['original']}'")
                
                # Count matches in top 5
                top5_activities = [p['activity'] for p in pred_items]
                matches_in_top5 = len(set(top5_activities) & set(gt_normalized))
                print(f"   🎯 Matches in top 5: {matches_in_top5}/{len(gt_normalized)}")
        
        # Compute metrics for different k values
        k_values = [1, 3, 5]
        metrics = {}
        
        total_evaluated = 0
        perfect_matches = 0
        top5_coverage = []
        
        for k in k_values:
            precision_scores = []
            recall_scores = []
            f1_scores = []
            
            for req_id in ground_truth:
                if req_id in matches_by_req:
                    # Get top-k predictions
                    predicted = [m['activity'] for m in matches_by_req[req_id][:k]]
                    
                    # Get ground truth (normalized)
                    actual = [item['normalized'] for item in ground_truth[req_id]]
                    
                    # Compute precision and recall
                    true_positives = len(set(predicted) & set(actual))
                    
                    precision = true_positives / len(predicted) if predicted else 0
                    recall = true_positives / len(actual) if actual else 0
                    
                    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                    
                    precision_scores.append(precision)
                    recall_scores.append(recall)
                    f1_scores.append(f1)
                    
                    if k == 1:
                        total_evaluated += 1
                        if precision == 1.0:
                            perfect_matches += 1
                    if k == 5:
                        top5_coverage.append(recall)
                else:
                    # No predictions for this requirement
                    precision_scores.append(0)
                    recall_scores.append(0)
                    f1_scores.append(0)
            
            metrics[f'precision_at_{k}'] = np.mean(precision_scores)
            metrics[f'recall_at_{k}'] = np.mean(recall_scores)
            metrics[f'f1_at_{k}'] = np.mean(f1_scores)
        
        # Overall coverage
        covered_requirements = len(set(matches_by_req.keys()) & set(ground_truth.keys()))
        total_requirements = len(ground_truth)
        metrics['coverage'] = covered_requirements / total_requirements if total_requirements > 0 else 0
        
        # Additional insights
        metrics['total_evaluated'] = total_evaluated
        metrics['perfect_matches'] = perfect_matches
        metrics['perfect_match_rate'] = perfect_matches / total_evaluated if total_evaluated > 0 else 0
        metrics['avg_top5_coverage'] = np.mean(top5_coverage) if top5_coverage else 0
        
        print(f"\n   📊 QUICK METRICS PREVIEW:")
        print(f"      F1@1: {metrics['f1_at_1']:.3f}")
        print(f"      F1@5: {metrics['f1_at_5']:.3f}")
        print(f"      Coverage: {metrics['coverage']:.1%}")
        print(f"      Perfect matches: {perfect_matches}/{total_evaluated}")
        
        return metrics
    
    def _analyze_results_enhanced(self, matches_df: pd.DataFrame, ground_truth: Dict, 
                                 requirements_df: pd.DataFrame, score_col: str, req_col: str, act_col: str) -> Dict:
        """Enhanced results analysis."""
        
        analysis = {
            'total_matches': len(matches_df),
            'unique_requirements': len(matches_df[req_col].unique()),
            'avg_matches_per_req': len(matches_df) / len(matches_df[req_col].unique()) if len(matches_df) > 0 else 0
        }
        
        # Score distribution if available
        if score_col and score_col in matches_df.columns:
            scores = matches_df[score_col]
            analysis['score_distribution'] = {
                'mean': float(scores.mean()),
                'min': float(scores.min()),
                'max': float(scores.max()),
                'std': float(scores.std()),
                'excellent': int(len(scores[scores >= 0.8])),
                'good': int(len(scores[scores >= 0.6])),
                'moderate': int(len(scores[scores >= 0.4]))
            }
        
        return analysis
    
    def _generate_report_enhanced(self, metrics: Dict, analysis: Dict) -> str:
        """Generate enhanced evaluation report."""
        
        report = f"""
ENHANCED MATCHING EVALUATION REPORT
{'='*50}

📊 METRICS SUMMARY
{'-'*20}
Coverage: {metrics.get('coverage', 0):.1%} of requirements have matches
Perfect Matches: {metrics.get('perfect_matches', 0)}/{metrics.get('total_evaluated', 0)} ({metrics.get('perfect_match_rate', 0):.1%})

Precision@1: {metrics.get('precision_at_1', 0):.3f}
Recall@1:    {metrics.get('recall_at_1', 0):.3f}
F1@1:        {metrics.get('f1_at_1', 0):.3f}

Precision@5: {metrics.get('precision_at_5', 0):.3f}
Recall@5:    {metrics.get('recall_at_5', 0):.3f}
F1@5:        {metrics.get('f1_at_5', 0):.3f}

📈 DATASET OVERVIEW
{'-'*20}
Total matches: {analysis.get('total_matches', 0)}
Unique requirements: {analysis.get('unique_requirements', 0)}
Avg matches per req: {analysis.get('avg_matches_per_req', 0):.1f}
Average top-5 coverage: {metrics.get('avg_top5_coverage', 0):.1%}
"""
        
        # Score distribution if available
        if 'score_distribution' in analysis:
            dist = analysis['score_distribution']
            report += f"""
🎯 SCORE DISTRIBUTION
{'-'*20}
Average score: {dist.get('mean', 0):.3f}
Score range: {dist.get('min', 0):.3f} - {dist.get('max', 0):.3f}
Excellent (≥0.8): {dist.get('excellent', 0)}
Good (≥0.6): {dist.get('good', 0)}
Moderate (≥0.4): {dist.get('moderate', 0)}
"""
        
        # Overall assessment
        f1_5 = metrics.get('f1_at_5', 0)
        coverage = metrics.get('coverage', 0)
        perfect_rate = metrics.get('perfect_match_rate', 0)
        
        report += f"""
🎯 OVERALL ASSESSMENT
{'-'*20}"""
        
        if f1_5 >= 0.7 and coverage >= 0.8:
            report += "\n🚀 EXCELLENT: Strong matching performance"
        elif f1_5 >= 0.5 and coverage >= 0.6:
            report += "\n✅ GOOD: Solid matching with room for improvement"
        elif f1_5 >= 0.3:
            report += "\n📈 MODERATE: Acceptable but needs tuning"
        else:
            report += "\n🔧 NEEDS WORK: Consider algorithm improvements"
        
        report += f"\n\nKey insights:"
        report += f"\n• F1@5 = {f1_5:.3f}, Coverage = {coverage:.1%}"
        report += f"\n• Perfect match rate = {perfect_rate:.1%}"
        report += f"\n• Fixed column mapping and normalization issues"
        
        return report
    
    def _save_results_enhanced(self):
        """Save enhanced results."""
        if UTILS_AVAILABLE and self.repo_manager:
            output_dir = self.repo_manager.structure['evaluation_results']
        else:
            output_dir = Path("outputs/evaluation_results")
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save metrics
        metrics_file = output_dir / "fixed_simple_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(self.results['metrics'], f, indent=2)
        
        # Save report with proper UTF-8 encoding
        report_file = output_dir / "fixed_simple_evaluation_report.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(self.results['report'])
        
        print(f"\n📁 Results saved:")
        print(f"   📊 Metrics: {metrics_file}")
        print(f"   📝 Report: {report_file}")


def main():
    """Run the fixed evaluation."""
    print("🚀 RUNNING FIXED SIMPLE EVALUATION")
    print("=" * 50)
    print("Goal: Fix column mapping, normalization, and debug issues")
    
    evaluator = FixedSimpleEvaluator()
    
    try:
        results = evaluator.evaluate_matches(
            matches_file="outputs/matching_results/aerospace_matches.csv",
            ground_truth_file="manual_matches.csv", 
            requirements_file="requirements.csv"
        )
        
        if "error" not in results:
            print(f"\n✅ Fixed evaluation complete!")
            print(f"📊 F1@5: {results['metrics']['f1_at_5']:.3f}")
            print(f"📈 Coverage: {results['metrics']['coverage']:.1%}")
            print(f"🎯 Perfect matches: {results['metrics']['perfect_matches']}/{results['metrics']['total_evaluated']}")
            
            # Show the debugging info that was used
            debug_info = results.get('debugging_info', {})
            print(f"\n🔧 DEBUG INFO:")
            print(f"   Score column: {debug_info.get('score_column')}")
            print(f"   Req column: {debug_info.get('req_column')}")
            print(f"   Activity column: {debug_info.get('activity_column')}")
            if 'file_paths' in debug_info:
                print(f"   File paths used: {debug_info['file_paths']}")
        else:
            print(f"\n❌ Evaluation failed: {results['error']}")
            
    except Exception as e:
        print(f"\n❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()