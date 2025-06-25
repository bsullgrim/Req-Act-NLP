#!/usr/bin/env python3
"""
Quick Debug Evaluation
======================

Minimal script to debug exactly what's happening with P.9 evaluation.
Uses proper repository structure and path resolution.
"""

import pandas as pd
import re
import sys
from pathlib import Path
from collections import defaultdict

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

def get_file_paths():
    """Get proper file paths using repository utilities."""
    
    if UTILS_AVAILABLE:
        repo_manager = RepositoryStructureManager("outputs")
        path_resolver = SmartPathResolver(repo_manager)
        
        # Define file mapping
        file_mapping = {
            'requirements': 'requirements.csv',
            'activities': 'activities.csv',
            'manual_matches': 'manual_matches.csv'
        }
        
        # Resolve paths
        resolved_paths = path_resolver.resolve_input_files(file_mapping)
        
        return {
            'matcher_results': "outputs/matching_results/aerospace_matches.csv",
            'ground_truth': resolved_paths['manual_matches'],
            'requirements': resolved_paths['requirements'],
            'activities': resolved_paths['activities']
        }
    else:
        # Fallback to manual paths
        return {
            'matcher_results': "outputs/matching_results/aerospace_matches.csv",
            'ground_truth': "data/raw/manual_matches.csv",
            'requirements': "data/raw/requirements.csv",
            'activities': "data/raw/activities.csv"
        }

def normalize_activity(activity):
    """Normalize activity name for comparison."""
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

def debug_p9_evaluation():
    """Debug P.9 evaluation specifically."""
    
    print("🔍 DEBUGGING P.9 EVALUATION")
    print("=" * 40)
    
    # Get proper file paths
    file_paths = get_file_paths()
    
    # 1. Load ground truth for P.9
    print("\n1. Loading ground truth...")
    try:
        gt_df = pd.read_csv(file_paths['ground_truth'])
        p9_row = gt_df[gt_df['ID'] == 'P.9'].iloc[0]
        
        satisfied_by = p9_row['Satisfied By']
        print(f"   Raw satisfied by: '{satisfied_by}'")
        
        # Parse ground truth
        activities = [activity.strip() for activity in satisfied_by.split(',')]
        normalized_gt = [normalize_activity(activity) for activity in activities]
        
        print(f"   Ground truth activities: {len(normalized_gt)}")
        for i, (orig, norm) in enumerate(zip(activities, normalized_gt)):
            print(f"      {i+1}. '{norm}' (from: '{orig[:50]}...')")
        
    except Exception as e:
        print(f"   ❌ Error loading ground truth: {e}")
        return
    
    # 2. Load matcher results for P.9
    print("\n2. Loading matcher results...")
    try:
        matches_df = pd.read_csv(file_paths['matcher_results'])
        print(f"   Loaded {len(matches_df)} total matches")
        print(f"   Columns: {list(matches_df.columns)}")
        
        # Find P.9 matches
        p9_matches = matches_df[matches_df['Requirement_ID'] == 'P.9']
        print(f"   P.9 matches found: {len(p9_matches)}")
        
        if len(p9_matches) == 0:
            print("   ❌ No P.9 matches found!")
            print("   Available requirement IDs:")
            unique_ids = matches_df['Requirement_ID'].unique()[:10]
            print(f"      {list(unique_ids)}")
            return
        
        # Show top matches for P.9
        print(f"   Top P.9 matches:")
        for i, (_, row) in enumerate(p9_matches.head().iterrows()):
            activity_name = row['Activity_Name']
            score = row.get('Combined_Score', 0)
            normalized_pred = normalize_activity(activity_name)
            
            # Check if it matches ground truth
            is_match = normalized_pred in normalized_gt
            match_status = "✅ MATCH" if is_match else "❌ MISS"
            
            print(f"      {i+1}. '{normalized_pred}' (score: {score:.3f}) {match_status}")
            print(f"         Original: '{activity_name}'")
        
    except Exception as e:
        print(f"   ❌ Error loading matcher results: {e}")
        return
    
    # 3. Manual evaluation calculation
    print("\n3. Manual evaluation calculation...")
    
    # Get top 5 predictions
    top5_matches = p9_matches.head(5)
    predicted_activities = [normalize_activity(row['Activity_Name']) for _, row in top5_matches.iterrows()]
    
    print(f"   Top 5 predictions: {predicted_activities}")
    print(f"   Ground truth: {normalized_gt}")
    
    # Calculate metrics
    true_positives = len(set(predicted_activities) & set(normalized_gt))
    precision = true_positives / len(predicted_activities) if predicted_activities else 0
    recall = true_positives / len(normalized_gt) if normalized_gt else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"   True positives: {true_positives}")
    print(f"   Precision@5: {precision:.3f}")
    print(f"   Recall@5: {recall:.3f}")
    print(f"   F1@5: {f1:.3f}")
    
    # Show which ones matched
    matches_found = set(predicted_activities) & set(normalized_gt)
    misses = set(normalized_gt) - set(predicted_activities)
    
    if matches_found:
        print(f"   ✅ Correctly found: {list(matches_found)}")
    if misses:
        print(f"   ❌ Missed: {list(misses)}")

def debug_overall_evaluation():
    """Quick overall evaluation debug."""
    
    print("\n🔍 OVERALL EVALUATION DEBUG")
    print("=" * 40)
    
    file_paths = get_file_paths()
    
    try:
        # Load data
        matches_df = pd.read_csv(file_paths['matcher_results'])
        gt_df = pd.read_csv(file_paths['ground_truth'])
        
        print(f"Matches shape: {matches_df.shape}")
        print(f"Ground truth shape: {gt_df.shape}")
        
        # Group matches by requirement
        matches_by_req = defaultdict(list)
        
        for _, row in matches_df.iterrows():
            req_id = str(row['Requirement_ID']).strip()
            activity = normalize_activity(row['Activity_Name'])
            score = row.get('Combined_Score', 0)
            
            matches_by_req[req_id].append({
                'activity': activity,
                'score': float(score) if pd.notna(score) else 0.0
            })
        
        # Sort by score
        for req_id in matches_by_req:
            matches_by_req[req_id].sort(key=lambda x: x['score'], reverse=True)
        
        # Load ground truth
        ground_truth = defaultdict(list)
        
        for _, row in gt_df.iterrows():
            req_id = str(row['ID']).strip()
            satisfied_by = str(row.get('Satisfied By', '')).strip()
            
            if satisfied_by and satisfied_by.lower() != 'nan':
                activities = [activity.strip() for activity in satisfied_by.split(',')]
                
                for activity in activities:
                    if activity:
                        normalized_activity = normalize_activity(activity)
                        if normalized_activity:
                            ground_truth[req_id].append(normalized_activity)
        
        print(f"Requirements with matches: {len(matches_by_req)}")
        print(f"Requirements with ground truth: {len(ground_truth)}")
        
        # Check overlap
        overlap = set(matches_by_req.keys()) & set(ground_truth.keys())
        print(f"Overlapping requirements: {len(overlap)}")
        
        if len(overlap) == 0:
            print("❌ CRITICAL: No overlapping requirements!")
            print(f"Sample matcher IDs: {list(matches_by_req.keys())[:5]}")
            print(f"Sample GT IDs: {list(ground_truth.keys())[:5]}")
        else:
            # Quick F1@5 calculation
            f1_scores = []
            
            for req_id in overlap:
                predicted = [m['activity'] for m in matches_by_req[req_id][:5]]
                actual = ground_truth[req_id]
                
                true_positives = len(set(predicted) & set(actual))
                precision = true_positives / len(predicted) if predicted else 0
                recall = true_positives / len(actual) if actual else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                
                f1_scores.append(f1)
            
            avg_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0
            print(f"Quick F1@5 calculation: {avg_f1:.3f}")
        
    except Exception as e:
        print(f"❌ Error in overall debug: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Run debug evaluation."""
    debug_p9_evaluation()
    debug_overall_evaluation()

if __name__ == "__main__":
    main()