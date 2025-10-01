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
from typing import Dict, List, Tuple, Optional, Any
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
                        requirements_file: str = "requirements.csv",
                        run_params: Optional[Dict[str, Any]] = None) -> Dict:
        """Run enhanced evaluation with debugging."""
        
        print("🔍 ENHANCED SIMPLE MATCHING EVALUATION")
        print("=" * 50)
        print("🐛 Debug mode: ON - Will show detailed diagnostics")
        
        # Get proper file paths
        file_paths = self.get_file_paths(matches_file, ground_truth_file, requirements_file)
        
        # Collect run metadata (now includes run_params)
        metadata = self._collect_run_metadata(file_paths, run_params)
        
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
        report = self._generate_report_enhanced(metrics, analysis, metadata)
        
        self.results = {
            'metrics': metrics,
            'analysis': analysis,
            'report': report,
            'metadata': metadata,
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
        
        # Compute ORIGINAL metrics for different k values
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
        
        # ADD NEW: Compute supplementary metrics
        supplementary = self._compute_supplementary_metrics(matches_by_req, ground_truth)
        metrics.update(supplementary)  # Add to main metrics dict
        
        print(f"\n   📊 QUICK METRICS PREVIEW:")
        print(f"      F1@1: {metrics['f1_at_1']:.3f}")
        print(f"      F1@5: {metrics['f1_at_5']:.3f}")
        print(f"      Coverage: {metrics['coverage']:.1%}")
        print(f"      Perfect matches: {perfect_matches}/{total_evaluated}")
        
        return metrics

    def _compute_supplementary_metrics(self, matches_by_req: dict, ground_truth: dict) -> dict:
        """Compute supplementary metrics for practical insight."""
        
        # Initialize tracking
        hit_at_k = {1: 0, 3: 0, 5: 0}
        success_at_1 = 0
        reciprocal_ranks = []
        total_requirements = len(ground_truth)
        
        print(f"   📊 Computing supplementary metrics...")
        
        for req_id in ground_truth:
            gt_normalized = [item['normalized'] for item in ground_truth[req_id]]
            
            if req_id in matches_by_req:
                # Get predictions (already sorted by score)
                predictions = [m['activity'] for m in matches_by_req[req_id]]
                
                # Find rank of first correct prediction
                first_correct_rank = None
                for i, pred in enumerate(predictions):
                    if pred in gt_normalized:
                        first_correct_rank = i + 1  # 1-indexed rank
                        break
                
                # Hit@K: Does top-K contain at least one correct answer?
                for k in [1, 3, 5]:
                    top_k_preds = predictions[:k]
                    if any(pred in gt_normalized for pred in top_k_preds):
                        hit_at_k[k] += 1
                
                # Success@1: Is top prediction correct?
                if len(predictions) > 0 and predictions[0] in gt_normalized:
                    success_at_1 += 1
                
                # MRR: Reciprocal of rank of first correct answer
                if first_correct_rank is not None:
                    reciprocal_ranks.append(1.0 / first_correct_rank)
                else:
                    reciprocal_ranks.append(0.0)  # No correct answer found
            else:
                # No predictions for this requirement
                reciprocal_ranks.append(0.0)
        
        # Calculate final metrics
        supplementary = {
            'hit_at_1': hit_at_k[1] / total_requirements,
            'hit_at_3': hit_at_k[3] / total_requirements, 
            'hit_at_5': hit_at_k[5] / total_requirements,
            'success_at_1': success_at_1 / total_requirements,
            'mrr': np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0,
            'total_requirements_evaluated': total_requirements
        }
        
        print(f"   📊 Supplementary metrics preview:")
        print(f"      Hit@5: {supplementary['hit_at_5']:.1%}")
        print(f"      Success@1: {supplementary['success_at_1']:.1%}")
        print(f"      MRR: {supplementary['mrr']:.3f}")
        
        return supplementary    

    def _collect_run_metadata(self, file_paths: dict, run_params: Optional[Dict[str, Any]] = None) -> dict:
        """Collect comprehensive metadata about the current run including all tuning parameters."""
        from datetime import datetime
        import platform
        import sys
        import re
        
        metadata = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'evaluator_version': 'Enhanced Simple Evaluator v2.0',
            'python_version': sys.version.split()[0],
            'platform': platform.platform(),
            'file_paths': file_paths
        }
        
        # PRIORITY 1: Use run parameters passed directly from matcher (most reliable)
        if run_params:
            print(f"   ✅ Using provided run parameters from matcher")
            
            # Core algorithm parameters
            metadata['algorithm_parameters'] = {
                'min_similarity_threshold': run_params.get('min_similarity', 'Unknown'),
                'top_n_matches': run_params.get('top_n', 'Unknown'),
                'output_file': run_params.get('output_file', 'Unknown'),
                'requirements_file': run_params.get('requirements_file', 'Unknown'),
                'activities_file': run_params.get('activities_file', 'Unknown'),
                'save_explanations': run_params.get('save_explanations', 'Unknown')
            }
            
            # Score component weights
            if 'weights' in run_params and run_params['weights']:
                metadata['score_weights'] = run_params['weights'].copy()
            
            # Additional matcher info if provided
            if 'spacy_model' in run_params:
                metadata['spacy_model'] = run_params['spacy_model']
            if 'domain_terms_count' in run_params:
                metadata['domain_terms_count'] = run_params['domain_terms_count']
            if 'synonyms_count' in run_params:
                metadata['synonyms_count'] = run_params['synonyms_count']
        
        # PRIORITY 2: Extract detailed matcher parameters from explanations file
        explanations_file = file_paths.get('matches', '').replace('.csv', '_explanations.json')
        if Path(explanations_file).exists():
            try:
                with open(explanations_file, 'r') as f:
                    explanations_data = json.load(f)
                    if explanations_data and len(explanations_data) > 0:
                        sample = explanations_data[0]
                        
                        # Extract detailed scoring information
                        metadata['matcher_info'] = {
                            'has_explanations': True,
                            'explanation_keys': list(sample.get('explanations', {}).keys()),
                            'score_components': list(sample.get('scores', {}).keys()),
                            'sample_explanations': {}
                        }
                        
                        # Parse explanations for parameter hints
                        explanations = sample.get('explanations', {})
                        
                        # Extract BM25 parameters from explanation text
                        if 'bm25' in explanations:
                            bm25_text = explanations['bm25']
                            metadata['matcher_info']['bm25_details'] = bm25_text
                            
                            # Try to extract k1 and b parameters from explanation
                            k1_match = re.search(r'k1[=:\s]*([0-9.]+)', bm25_text)
                            if k1_match:
                                if 'algorithm_parameters' not in metadata:
                                    metadata['algorithm_parameters'] = {}
                                metadata['algorithm_parameters']['bm25_k1'] = float(k1_match.group(1))
                            
                            b_match = re.search(r'\bb[=:\s]*([0-9.]+)', bm25_text)
                            if b_match:
                                if 'algorithm_parameters' not in metadata:
                                    metadata['algorithm_parameters'] = {}
                                metadata['algorithm_parameters']['bm25_b'] = float(b_match.group(1))
                            
                        # Extract semantic model info
                        if 'semantic' in explanations:
                            semantic_text = explanations['semantic']
                            metadata['matcher_info']['semantic_details'] = semantic_text
                            if isinstance(semantic_text, str):
                                if 'spacy' in semantic_text.lower():
                                    metadata['semantic_method'] = 'spaCy'
                                elif 'transformer' in semantic_text.lower():
                                    metadata['semantic_method'] = 'Transformer'
                                elif 'sentence' in semantic_text.lower():
                                    metadata['semantic_method'] = 'SentenceTransformer'
                            else:
                                # If it's a dict, optionally store keys or a summary
                                metadata['semantic_method'] = 'Unknown (dict format)'
                            
                        # Extract domain scoring details
                        if 'domain' in explanations:
                            domain_text = explanations['domain']
                            metadata['matcher_info']['domain_details'] = domain_text
                            if isinstance(domain_text, str):
                                domain_terms_mentioned = len(re.findall(r'aerospace|domain|weight', domain_text.lower()))
                            else:
                                # For dicts, count the keys you care about
                                domain_terms_mentioned = len(domain_text.keys()) if isinstance(domain_text, dict) else 0
                            metadata['domain_terms_in_explanation'] = domain_terms_mentioned
                                                        
            except Exception as e:
                print(f"   ⚠️ Could not parse explanations file: {e}")
        
        # PRIORITY 3: Try to extract parameters from the matcher source files
        matcher_files = ['matcher.py', 'src/matching/matcher.py', '../matcher.py', './matcher.py']
        for matcher_file in matcher_files:
            if Path(matcher_file).exists():
                try:
                    with open(matcher_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                        
                    # Extract key parameters using regex
                    extracted_params = {}
                    
                    # Look for BM25 k1 parameter
                    k1_matches = re.findall(r'k1\s*[=:]\s*([0-9.]+)', content)
                    if k1_matches:
                        extracted_params['bm25_k1_from_source'] = float(k1_matches[-1])
                    
                    # Look for BM25 b parameter  
                    b_matches = re.findall(r'\bb\s*[=:]\s*([0-9.]+)', content)
                    if b_matches:
                        extracted_params['bm25_b_from_source'] = float(b_matches[-1])
                    
                    # Look for similarity thresholds
                    min_sim_matches = re.findall(r'min_similarity[^=]*[=:]\s*([0-9.]+)', content)
                    if min_sim_matches:
                        extracted_params['min_similarity_from_source'] = float(min_sim_matches[-1])
                    
                    # Look for default weights
                    weights_sections = re.findall(r'weights\s*[=:]\s*\{([^}]+)\}', content, re.MULTILINE | re.DOTALL)
                    for weights_text in weights_sections:
                        weight_matches = re.findall(r"['\"]?(\w+)['\"]?\s*:\s*([0-9.]+)", weights_text)
                        if weight_matches:
                            extracted_params['default_weights_from_source'] = {name: float(value) for name, value in weight_matches}
                    
                    # Look for default constants
                    default_matches = re.findall(r'DEFAULT_(\w+)\s*=\s*([0-9.]+)', content)
                    for const_name, const_value in default_matches:
                        extracted_params[f'default_{const_name.lower()}'] = float(const_value)
                    
                    # Look for class names
                    class_matches = re.findall(r'class\s+(\w*Matcher\w*)', content)
                    if class_matches:
                        extracted_params['matcher_class_from_source'] = class_matches[0]
                    
                    if extracted_params:
                        metadata['source_code_analysis'] = extracted_params
                        metadata['matcher_source_file'] = matcher_file
                        print(f"   ✅ Extracted parameters from {matcher_file}")
                        break
                        
                except Exception as e:
                    print(f"   ⚠️ Could not read {matcher_file}: {e}")
                    continue
        
        # PRIORITY 4: Analyze the matches file for additional insights
        try:
            matches_df = pd.read_csv(file_paths.get('matches', ''))
            if not matches_df.empty:
                score_cols = [col for col in matches_df.columns if 'score' in col.lower()]
                
                # Basic match analysis
                metadata['match_analysis'] = {
                    'total_matches_generated': len(matches_df),
                    'score_columns': score_cols,
                    'has_domain_score': 'Domain_Score' in matches_df.columns,
                    'has_semantic_score': 'Semantic_Score' in matches_df.columns,
                    'has_bm25_score': 'BM25_Score' in matches_df.columns,
                    'has_query_expansion_score': 'Query_Expansion_Score' in matches_df.columns,
                }
                
                # Requirement analysis
                if 'Requirement_ID' in matches_df.columns:
                    unique_reqs = len(matches_df['Requirement_ID'].unique())
                    avg_matches = len(matches_df) / unique_reqs
                    metadata['match_analysis'].update({
                        'unique_requirements_matched': unique_reqs,
                        'avg_matches_per_requirement': avg_matches,
                        'inferred_top_n': int(matches_df['Requirement_ID'].value_counts().mode()[0]) if len(matches_df) > 0 else None
                    })
                
                # Score distribution analysis
                score_analysis = {}
                for col in score_cols:
                    if col in matches_df.columns:
                        scores = matches_df[col].dropna()
                        if len(scores) > 0:
                            score_analysis[col] = {
                                'min': float(scores.min()),
                                'max': float(scores.max()),
                                'mean': float(scores.mean()),
                                'std': float(scores.std()),
                                'median': float(scores.median()),
                                'count': int(len(scores)),
                                'non_zero_count': int(len(scores[scores > 0]))
                            }
                            
                            # Score distribution bins
                            if col == 'Combined_Score':
                                score_analysis[col]['distribution'] = {
                                    'excellent_count': int(len(scores[scores >= 0.8])),
                                    'good_count': int(len(scores[(scores >= 0.6) & (scores < 0.8)])),
                                    'moderate_count': int(len(scores[(scores >= 0.4) & (scores < 0.6)])),
                                    'poor_count': int(len(scores[scores < 0.4]))
                                }
                
                if score_analysis:
                    metadata['score_analysis'] = score_analysis
                    
            else:
                print(f"   ⚠️ Matches file is empty")
                
        except Exception as e:
            print(f"   ⚠️ Could not analyze matches file: {e}")
        
        # PRIORITY 5: Try to infer additional system information
        try:
            # Check if spaCy models are available
            try:
                import spacy
                available_models = []
                for model_name in ['en_core_web_sm', 'en_core_web_md', 'en_core_web_lg', 'en_core_web_trf']:
                    try:
                        spacy.load(model_name)
                        available_models.append(model_name)
                    except:
                        pass
                metadata['available_spacy_models'] = available_models
            except ImportError:
                metadata['available_spacy_models'] = []
            
            # Check for optional dependencies
            optional_deps = {}
            try:
                import sklearn
                optional_deps['sklearn'] = sklearn.__version__
            except ImportError:
                optional_deps['sklearn'] = 'Not installed'
            
            try:
                import sentence_transformers
                optional_deps['sentence_transformers'] = sentence_transformers.__version__
            except ImportError:
                optional_deps['sentence_transformers'] = 'Not installed'
                
            metadata['dependencies'] = optional_deps
            
        except Exception as e:
            print(f"   ⚠️ Could not collect system information: {e}")
        
        # Summary of what was collected
        collection_summary = []
        if run_params:
            collection_summary.append("Direct run parameters")
        if 'matcher_info' in metadata:
            collection_summary.append("Explanation file analysis")
        if 'source_code_analysis' in metadata:
            collection_summary.append("Source code analysis") 
        if 'match_analysis' in metadata:
            collection_summary.append("Match file analysis")
        if 'dependencies' in metadata:
            collection_summary.append("System information")
        
        metadata['collection_summary'] = collection_summary
        print(f"   📊 Collected metadata from: {', '.join(collection_summary)}")
        
        return metadata
    
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
    
    def _generate_report_enhanced(self, metrics: Dict, analysis: Dict, metadata: Dict = None) -> str:
        """Generate comprehensive evaluation report with all collected metadata."""
        
        report = f"""
    ENHANCED MATCHING EVALUATION REPORT
    {'='*50}

    ⚙️ RUN METADATA
    {'-'*15}
    Timestamp: {metadata.get('timestamp', 'Unknown') if metadata else 'Unknown'}
    Evaluator: {metadata.get('evaluator_version', 'Enhanced Simple Evaluator') if metadata else 'Enhanced Simple Evaluator'}
    Python: {metadata.get('python_version', 'Unknown') if metadata else 'Unknown'}
    Platform: {metadata.get('platform', 'Unknown') if metadata else 'Unknown'}
    Data Collection: {', '.join(metadata.get('collection_summary', ['Unknown'])) if metadata else 'Unknown'}
    """
        
        # SECTION 1: Algorithm Parameters (Direct from Run)
        if metadata and 'algorithm_parameters' in metadata:
            params = metadata['algorithm_parameters']
            report += f"""
    🔧 ALGORITHM PARAMETERS
    {'-'*23}
    Similarity Threshold: {params.get('min_similarity_threshold', 'Unknown')}
    Top N Matches: {params.get('top_n_matches', 'Unknown')}
    Output File: {params.get('output_file', 'Unknown')}
    Requirements File: {params.get('requirements_file', 'Unknown')}
    Activities File: {params.get('activities_file', 'Unknown')}
    Save Explanations: {params.get('save_explanations', 'Unknown')}"""
            
            # BM25 parameters if available
            if 'bm25_k1' in params or 'bm25_b' in params:
                report += f"""

    BM25 Parameters:
    k1 (term frequency saturation): {params.get('bm25_k1', 'Unknown')}
    b (length normalization): {params.get('bm25_b', 'Unknown')}"""
        
        # SECTION 2: Score Component Weights
        if metadata and 'score_weights' in metadata:
            weights = metadata['score_weights']
            report += f"""
            
    ⚖️ SCORE COMPONENT WEIGHTS
    {'-'*26}"""
            total_weight = sum(weights.values()) if weights else 1
            for component, weight in weights.items():
                component_name = component.replace('_', ' ').title()
                percentage = (weight / total_weight) * 100 if total_weight > 0 else 0
                report += f"""
    {component_name}: {weight} ({percentage:.1f}%)"""
            
            report += f"""
    Total Weight: {total_weight}"""
        
        # SECTION 3: Matcher Configuration Details
        if metadata and 'matcher_info' in metadata:
            matcher_info = metadata['matcher_info']
            report += f"""

    🤖 MATCHER CONFIGURATION
    {'-'*25}
    Has Explanations: {'✅ Yes' if matcher_info.get('has_explanations') else '❌ No'}
    Score Components: {', '.join(matcher_info.get('score_components', ['Unknown']))}
    Explanation Types: {', '.join(matcher_info.get('explanation_keys', ['Unknown']))}"""
            
            # Semantic method detection
            if metadata.get('semantic_method'):
                report += f"""
    Semantic Method: {metadata['semantic_method']}"""
            
            # spaCy model info
            if metadata.get('spacy_model'):
                report += f"""
    spaCy Model: {metadata['spacy_model']}"""
        
        # SECTION 4: Match Generation Analysis
        if metadata and 'match_analysis' in metadata:
            match_info = metadata['match_analysis']
            report += f"""

    📊 MATCH GENERATION ANALYSIS
    {'-'*29}
    Total Matches Generated: {match_info.get('total_matches_generated', 'Unknown')}
    Unique Requirements Matched: {match_info.get('unique_requirements_matched', 'Unknown')}"""
    # Avg Matches per Requirement: {match_info.get('avg_matches_per_requirement', 'Unknown'):.1f if isinstance(match_info.get('avg_matches_per_requirement'), (int, float)) else 'Unknown'}"""
            
            if match_info.get('inferred_top_n'):
                report += f"""
    Inferred Top-N Setting: {match_info['inferred_top_n']}"""
            
            # Feature availability
            features = []
            if match_info.get('has_domain_score'):
                features.append('Domain Scoring')
            if match_info.get('has_semantic_score'):
                features.append('Semantic Similarity')
            if match_info.get('has_bm25_score'):
                features.append('BM25 Scoring')
            if match_info.get('has_query_expansion_score'):
                features.append('Query Expansion')
            
            if features:
                report += f"""
    Active Features: {', '.join(features)}"""
        
        # SECTION 5: Detailed Score Analysis
        if metadata and 'score_analysis' in metadata:
            score_analysis = metadata['score_analysis']
            report += f"""

    📈 SCORE COMPONENT ANALYSIS
    {'-'*28}"""
            
            for component, stats in score_analysis.items():
                component_name = component.replace('_', ' ').title()
                report += f"""

    {component_name}:
    Range: {stats['min']:.3f} - {stats['max']:.3f}
    Mean: {stats['mean']:.3f} (±{stats['std']:.3f})
    Median: {stats['median']:.3f}
    Non-zero: {stats['non_zero_count']}/{stats['count']} ({stats['non_zero_count']/stats['count']*100:.1f}%)"""
                
                # Special distribution analysis for Combined Score
                if component == 'Combined_Score' and 'distribution' in stats:
                    dist = stats['distribution']
                    total = sum(dist.values())
                    report += f"""
    Quality Distribution:
        Excellent (≥0.8): {dist['excellent_count']} ({dist['excellent_count']/total*100:.1f}%)
        Good (0.6-0.8): {dist['good_count']} ({dist['good_count']/total*100:.1f}%)
        Moderate (0.4-0.6): {dist['moderate_count']} ({dist['moderate_count']/total*100:.1f}%)
        Poor (<0.4): {dist['poor_count']} ({dist['poor_count']/total*100:.1f}%)"""
        
        # SECTION 6: System Environment
        if metadata and ('dependencies' in metadata or 'available_spacy_models' in metadata):
            report += f"""

    🖥️ SYSTEM ENVIRONMENT
    {'-'*20}"""
            
            if 'dependencies' in metadata:
                deps = metadata['dependencies']
                report += f"""
    Dependencies:
    scikit-learn: {deps.get('sklearn', 'Unknown')}
    sentence-transformers: {deps.get('sentence_transformers', 'Unknown')}"""
            
            if 'available_spacy_models' in metadata:
                models = metadata['available_spacy_models']
                if models:
                    report += f"""
    Available spaCy Models: {', '.join(models)}"""
                else:
                    report += f"""
    Available spaCy Models: None detected"""
        
        # SECTION 7: Source Code Analysis (if available)
        if metadata and 'source_code_analysis' in metadata:
            source_analysis = metadata['source_code_analysis']
            report += f"""

    🔍 SOURCE CODE ANALYSIS
    {'-'*23}
    Source File: {metadata.get('matcher_source_file', 'Unknown')}"""
            
            if 'matcher_class_from_source' in source_analysis:
                report += f"""
    Matcher Class: {source_analysis['matcher_class_from_source']}"""
            
            if 'bm25_k1_from_source' in source_analysis or 'bm25_b_from_source' in source_analysis:
                report += f"""
    BM25 Parameters in Source:
    k1: {source_analysis.get('bm25_k1_from_source', 'Not found')}
    b: {source_analysis.get('bm25_b_from_source', 'Not found')}"""
            
            if 'default_weights_from_source' in source_analysis:
                default_weights = source_analysis['default_weights_from_source']
                report += f"""
    Default Weights in Source:"""
                for comp, weight in default_weights.items():
                    report += f"""
    {comp}: {weight}"""
        
        # SECTION 8: File Information
        if metadata and 'file_paths' in metadata:
            file_paths = metadata['file_paths']
            report += f"""

    📁 DATA FILES
    {'-'*12}
    Matches: {Path(file_paths.get('matches', 'Unknown')).name}
    Ground Truth: {Path(file_paths.get('ground_truth', 'Unknown')).name}
    Requirements: {Path(file_paths.get('requirements', 'Unknown')).name}"""
        
        # SECTION 9: Core Evaluation Metrics
        report += f"""

    📊 CORE METRICS (Multi-label Evaluation)
    {'-'*25}
    Coverage: {metrics.get('coverage', 0):.1%} of requirements have matches
    Perfect Matches: {metrics.get('perfect_matches', 0)}/{metrics.get('total_evaluated', 0)} ({metrics.get('perfect_match_rate', 0):.1%})

    Precision@1: {metrics.get('precision_at_1', 0):.3f}
    Recall@1:    {metrics.get('recall_at_1', 0):.3f}
    F1@1:        {metrics.get('f1_at_1', 0):.3f}

    Precision@3: {metrics.get('precision_at_3', 0):.3f}
    Recall@3:    {metrics.get('recall_at_3', 0):.3f}
    F1@3:        {metrics.get('f1_at_3', 0):.3f}

    Precision@5: {metrics.get('precision_at_5', 0):.3f}
    Recall@5:    {metrics.get('recall_at_5', 0):.3f}
    F1@5:        {metrics.get('f1_at_5', 0):.3f}"""
        
        # SECTION 10: Practical Metrics
        report += f"""

    🎯 PRACTICAL METRICS (Any-Correct Evaluation)  
    {'-'*30}
    Hit@1:       {metrics.get('hit_at_1', 0):.1%} (≥1 correct in top-1)
    Hit@3:       {metrics.get('hit_at_3', 0):.1%} (≥1 correct in top-3)
    Hit@5:       {metrics.get('hit_at_5', 0):.1%} (≥1 correct in top-5)
    Success@1:   {metrics.get('success_at_1', 0):.1%} (top prediction correct)
    Perfect Match Rate: {metrics.get('perfect_match_rate', 0):.1%} (top-1 precision = 1.0)
    MRR:         {metrics.get('mrr', 0):.3f} (mean reciprocal rank)"""
        
        # SECTION 11: Dataset Overview
        report += f"""

    📈 DATASET OVERVIEW
    {'-'*20}
    Total matches: {analysis.get('total_matches', 0)}
    Unique requirements: {analysis.get('unique_requirements', 0)}
    Avg matches per req: {analysis.get('avg_matches_per_req', 0):.1f}
    Average top-5 coverage: {metrics.get('avg_top5_coverage', 0):.1%}"""
        
        # SECTION 12: Score Distribution (from analysis)
        if 'score_distribution' in analysis:
            dist = analysis['score_distribution']
            report += f"""

    🎯 EVALUATION SCORE DISTRIBUTION
    {'-'*32}
    Average score: {dist.get('mean', 0):.3f}
    Score range: {dist.get('min', 0):.3f} - {dist.get('max', 0):.3f}
    Standard deviation: {dist.get('std', 0):.3f}
    Excellent (≥0.8): {dist.get('excellent', 0)}
    Good (≥0.6): {dist.get('good', 0)}
    Moderate (≥0.4): {dist.get('moderate', 0)}"""
        
        # SECTION 13: Performance Assessment
        f1_5 = metrics.get('f1_at_5', 0)
        hit_5 = metrics.get('hit_at_5', 0)
        success_1 = metrics.get('success_at_1', 0)
        coverage = metrics.get('coverage', 0)
        perfect_rate = metrics.get('perfect_match_rate', 0)
        
        report += f"""

    🎯 OVERALL ASSESSMENT
    {'-'*20}"""
        
        # Multi-label assessment
        if f1_5 >= 0.7 and coverage >= 0.8:
            report += "\n   🚀 MULTI-LABEL: Excellent - Strong performance finding all ground truth activities"
        elif f1_5 >= 0.5 and coverage >= 0.6:
            report += "\n   ✅ MULTI-LABEL: Good - Solid performance with room for improvement"
        elif f1_5 >= 0.3:
            report += "\n   📈 MULTI-LABEL: Moderate - Acceptable but needs parameter tuning"
        else:
            report += "\n   🔧 MULTI-LABEL: Needs work - Consider algorithm improvements"
        
        # Practical assessment
        if hit_5 >= 0.8:
            report += "\n   🎯 PRACTICAL: Excellent - Finds useful matches for most requirements"
        elif hit_5 >= 0.6:
            report += "\n   🎯 PRACTICAL: Good - Finds useful matches for majority of requirements"
        elif hit_5 >= 0.4:
            report += "\n   🎯 PRACTICAL: Moderate - Finds useful matches for some requirements"
        else:
            report += "\n   🎯 PRACTICAL: Poor - Struggles to find useful matches"
        
        # SECTION 14: Key Insights
        report += f"""

    📊 KEY INSIGHTS
    {'-'*15}
    • Multi-label F1@5 = {f1_5:.3f} (finds all ground truth activities)
    • Practical Hit@5 = {hit_5:.1%} (finds at least one good match)
    • Success@1 = {success_1:.1%} (perfect top prediction rate)
    • Perfect Match Rate = {perfect_rate:.1%} (requirements with top-1 precision = 1.0)
    • Coverage = {coverage:.1%} (requirements attempted)

    💡 INTERPRETATION
    {'-'*15}
    The gap between F1@5 ({f1_5:.3f}) and Hit@5 ({hit_5:.1%}) shows how much
    the system is penalized for not finding ALL ground truth activities."""
        
        if hit_5 > 0 and f1_5 > 0:
            gap_ratio = hit_5 / f1_5
            if gap_ratio > 2:
                report += "\n   Large gap suggests many requirements have multiple valid activities."
            elif gap_ratio > 1.5:
                report += "\n   Moderate gap suggests some requirements have multiple valid activities."
            else:
                report += "\n   Small gap suggests most requirements have single ground truth activities."
        
        # SECTION 15: Recommendations
        report += f"""

    🔧 PERFORMANCE RECOMMENDATIONS
    {'-'*30}"""
        
        recommendations = []
        
        if perfect_rate < 0.3:
            recommendations.append("    • Consider tuning similarity thresholds - low perfect match rate")
        if hit_5 > 0.8 and f1_5 < 0.5:
            recommendations.append("    • System finds good matches but struggles with multi-activity requirements")
        if success_1 < 0.4:
            recommendations.append("    • Consider improving ranking/scoring - top predictions often incorrect")
        if coverage < 0.95:
            recommendations.append("    • Consider lowering minimum similarity threshold for better coverage")
        
        # Score component recommendations
        if metadata and 'score_analysis' in metadata:
            score_analysis = metadata['score_analysis']
            if 'Semantic_Score' in score_analysis:
                semantic_stats = score_analysis['Semantic_Score']
                if semantic_stats['mean'] < 0.3:
                    recommendations.append("• Semantic scores are low - consider different semantic model")
            
            if 'Domain_Score' in score_analysis:
                domain_stats = score_analysis['Domain_Score']
                if domain_stats['non_zero_count'] / domain_stats['count'] < 0.5:
                    recommendations.append("• Many zero domain scores - check aerospace term coverage")
        
        # Dependency recommendations
        if metadata and 'dependencies' in metadata:
            deps = metadata['dependencies']
            if deps.get('sentence_transformers') == 'Not installed':
                recommendations.append("• Install sentence-transformers for enhanced semantic similarity")
            if deps.get('sklearn') == 'Not installed':
                recommendations.append("• Install scikit-learn for TF-IDF domain term extraction")
        
        if recommendations:
            for rec in recommendations:
                report += f"\n{rec}"
        else:
            report += "\n• System performance appears well-tuned for current requirements"
        
        # SECTION 16: Configuration Summary for Version Tracking
        report += f"""

    📋 CONFIGURATION SUMMARY (for version comparison)
    {'-'*48}"""
        
        config_hash_components = []
        if metadata and 'algorithm_parameters' in metadata:
            params = metadata['algorithm_parameters']
            config_hash_components.extend([
                f"threshold:{params.get('min_similarity_threshold', 'unk')}",
                f"topn:{params.get('top_n_matches', 'unk')}"
            ])
        
        if metadata and 'score_weights' in metadata:
            weights = metadata['score_weights']
            weights_str = '|'.join([f"{k}:{v}" for k, v in sorted(weights.items())])
            config_hash_components.append(f"weights:{weights_str}")
        
        config_signature = ' | '.join(config_hash_components) if config_hash_components else 'Unknown configuration'
        report += f"""
    Configuration Signature: {config_signature}
    Evaluation Date: {metadata.get('timestamp', 'Unknown') if metadata else 'Unknown'}
    F1@5 Score: {f1_5:.3f}
    Hit@5 Score: {hit_5:.1%}

    Copy this summary for version comparison tracking.
    """
        
        return report    
    
    def _save_results_enhanced(self):
        """Save enhanced results."""
        if UTILS_AVAILABLE and self.repo_manager:
            output_dir = self.repo_manager.structure['evaluation_results']
        else:
            output_dir = Path("outputs/evaluation_results")
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save metrics
        metrics_file = output_dir / "evaluation_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(self.results['metrics'], f, indent=2)
        
        # Save report with proper UTF-8 encoding
        report_file = output_dir / "matching_evaluation_report.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(self.results['report'])
        
        print(f"\n📁 Results saved:")
        print(f"   📊 Metrics: {metrics_file}")
        print(f"   📝 Report: {report_file}")

def main():
    """Run the fixed evaluation."""
    print("🚀 RUNNING ENHANCED SIMPLE EVALUATION")
    print("=" * 50)
    print("Goal: Multi-label + practical metrics for complete picture")
    
    evaluator = FixedSimpleEvaluator()
    
    try:
        results = evaluator.evaluate_matches(
            matches_file="outputs/matching_results/aerospace_matches.csv",
            ground_truth_file="manual_matches.csv", 
            requirements_file="requirements.csv"
        )
        
        if "error" not in results:
            print(f"\n✅ Enhanced evaluation complete!")
            
            # Core metrics
            print(f"\n📊 CORE METRICS (Multi-label):")
            print(f"   F1@1: {results['metrics']['f1_at_1']:.3f}")
            print(f"   F1@3: {results['metrics']['f1_at_3']:.3f}")
            print(f"   F1@5: {results['metrics']['f1_at_5']:.3f}")
            print(f"   Coverage: {results['metrics']['coverage']:.1%}")
            
            # Practical metrics
            print(f"\n🎯 PRACTICAL METRICS (Any-correct):")
            print(f"   Hit@1: {results['metrics']['hit_at_1']:.1%}")
            print(f"   Hit@3: {results['metrics']['hit_at_3']:.1%}")
            print(f"   Hit@5: {results['metrics']['hit_at_5']:.1%}")
            print(f"   Success@1: {results['metrics']['success_at_1']:.1%}")
            print(f"   MRR: {results['metrics']['mrr']:.3f}")
            
            # Quick insight
            f1_5 = results['metrics']['f1_at_5']
            hit_5 = results['metrics']['hit_at_5']
            print(f"\n💡 Gap Analysis:")
            print(f"   F1@5 vs Hit@5: {f1_5:.3f} vs {hit_5:.1%}")
            if hit_5 > f1_5 * 1.5:
                print(f"   → System finds good matches but many requirements have multiple ground truth activities")
            else:
                print(f"   → System performance is consistent across both evaluation approaches")
            
            # Show the debugging info that was used
            debug_info = results.get('debugging_info', {})
            print(f"\n🔧 DEBUG INFO:")
            print(f"   Score column: {debug_info.get('score_column')}")
            print(f"   Req column: {debug_info.get('req_column')}")
            print(f"   Activity column: {debug_info.get('activity_column')}")
            
        else:
            print(f"\n❌ Evaluation failed: {results['error']}")
            
    except Exception as e:
        print(f"\n❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()