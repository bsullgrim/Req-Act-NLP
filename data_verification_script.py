#!/usr/bin/env python3
"""
Data Verification Script
========================

Quick diagnostic script to verify data files before running evaluation.
Uses proper repository structure and path resolution.
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path
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

def verify_data_files():
    """Verify all data files and identify issues."""
    
    print("🔍 DATA VERIFICATION")
    print("=" * 50)
    
    # Setup paths properly
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
        
        files_to_check = [
            ("Matcher Results", "outputs/matching_results/aerospace_matches.csv"),
            ("Ground Truth", resolved_paths['manual_matches']),
            ("Requirements", resolved_paths['requirements']),
            ("Activities", resolved_paths['activities'])
        ]
    else:
        # Fallback to manual paths
        files_to_check = [
            ("Matcher Results", "outputs/matching_results/aerospace_matches.csv"),
            ("Ground Truth", "data/raw/manual_matches.csv"),
            ("Requirements", "data/raw/requirements.csv"),
            ("Activities", "data/raw/activities.csv")
        ]
    
    file_status = {}
    
    for name, filepath in files_to_check:
        print(f"\n📁 Checking {name}: {filepath}")
        status = check_file(filepath)
        file_status[name] = status
        
        if status['exists']:
            print(f"   ✅ File found ({status['size']} bytes)")
            print(f"   📋 Columns: {status['columns']}")
            print(f"   📏 Shape: {status['shape']}")
            
            if status['encoding_issues']:
                print(f"   ⚠️  Encoding: {status['encoding']}")
            
            if name == "Matcher Results":
                analyze_matcher_results(status['data'])
            elif name == "Ground Truth":
                analyze_ground_truth(status['data'])
                
        else:
            print(f"   ❌ File not found!")
    
    # Summary
    print(f"\n📊 VERIFICATION SUMMARY")
    print("-" * 30)
    
    critical_files = ["Matcher Results", "Ground Truth"]
    missing_critical = [name for name in critical_files if not file_status.get(name, {}).get('exists', False)]
    
    if missing_critical:
        print(f"❌ CRITICAL: Missing files: {missing_critical}")
        print("   Cannot proceed with evaluation.")
        return False
    else:
        print("✅ All critical files found.")
        
        # Check for potential issues
        matcher_data = file_status.get("Matcher Results", {}).get('data')
        gt_data = file_status.get("Ground Truth", {}).get('data')
        
        if matcher_data is not None and gt_data is not None:
            check_compatibility(matcher_data, gt_data)
        
        return True

def check_file(filepath):
    """Check individual file and return status."""
    status = {
        'exists': False,
        'size': 0,
        'encoding': 'unknown',
        'encoding_issues': False,
        'columns': [],
        'shape': (0, 0),
        'data': None
    }
    
    path = Path(filepath)
    if not path.exists():
        return status
    
    status['exists'] = True
    status['size'] = path.stat().st_size
    
    # Try different encodings
    encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252']
    data = None
    
    for encoding in encodings:
        try:
            data = pd.read_csv(filepath, encoding=encoding)
            status['encoding'] = encoding
            if encoding != 'utf-8':
                status['encoding_issues'] = True
            break
        except (UnicodeDecodeError, pd.errors.EmptyDataError):
            continue
    
    if data is not None:
        status['data'] = data
        status['columns'] = list(data.columns)
        status['shape'] = data.shape
    
    return status

def analyze_matcher_results(df):
    """Analyze matcher results for common issues."""
    print(f"   🔍 Matcher Results Analysis:")
    
    # Check for score columns
    score_columns = [col for col in df.columns if 'score' in col.lower()]
    print(f"      Score columns found: {score_columns}")
    
    # Check the primary score column
    primary_score_col = None
    if 'Combined_Score' in df.columns:
        primary_score_col = 'Combined_Score'
    elif 'Combined Score' in df.columns:
        primary_score_col = 'Combined Score'
    elif score_columns:
        primary_score_col = score_columns[0]
    
    if primary_score_col:
        scores = df[primary_score_col]
        print(f"      Primary score column: '{primary_score_col}'")
        print(f"      Score range: {scores.min():.3f} - {scores.max():.3f}")
        print(f"      Score mean: {scores.mean():.3f}")
        print(f"      Zero scores: {(scores == 0).sum()}/{len(scores)}")
        
        if scores.max() == 0:
            print(f"      ⚠️  WARNING: All scores are zero!")
        elif scores.mean() < 0.1:
            print(f"      ⚠️  WARNING: Very low average scores!")
    else:
        print(f"      ❌ No score column found!")
    
    # Check requirement ID column
    req_id_columns = [col for col in df.columns if 'id' in col.lower() and 'requirement' in col.lower()]
    if not req_id_columns:
        req_id_columns = [col for col in df.columns if col.lower() in ['id', 'requirement_id', 'req_id']]
    
    if req_id_columns:
        req_col = req_id_columns[0]
        req_ids = df[req_col].unique()
        print(f"      Requirement ID column: '{req_col}'")
        print(f"      Unique requirements: {len(req_ids)}")
        print(f"      Sample IDs: {list(req_ids)[:5]}")
    else:
        print(f"      ⚠️  No clear requirement ID column found!")
    
    # Check activity column
    activity_columns = [col for col in df.columns if 'activity' in col.lower()]
    if activity_columns:
        act_col = activity_columns[0]
        print(f"      Activity column: '{act_col}'")
        sample_activities = df[act_col].head(3).tolist()
        print(f"      Sample activities: {sample_activities}")
    else:
        print(f"      ⚠️  No clear activity column found!")

def analyze_ground_truth(df):
    """Analyze ground truth for common issues."""
    print(f"   🔍 Ground Truth Analysis:")
    
    # Check columns
    print(f"      Columns: {list(df.columns)}")
    
    # Find key columns
    id_col = None
    satisfied_col = None
    
    for col in df.columns:
        if col.lower() in ['id', 'requirement_id', 'req_id']:
            id_col = col
        elif 'satisfied' in col.lower() or 'activity' in col.lower():
            satisfied_col = col
    
    if not id_col:
        id_col = df.columns[0]
        print(f"      ⚠️  Using first column as ID: '{id_col}'")
    else:
        print(f"      ID column: '{id_col}'")
    
    if not satisfied_col:
        satisfied_col = df.columns[-1]
        print(f"      ⚠️  Using last column as activities: '{satisfied_col}'")
    else:
        print(f"      Activities column: '{satisfied_col}'")
    
    # Analyze data
    req_ids = df[id_col].unique()
    print(f"      Requirements in ground truth: {len(req_ids)}")
    print(f"      Sample requirement IDs: {list(req_ids)[:5]}")
    
    # Check for empty/missing activities
    activities_col = df[satisfied_col]
    empty_activities = activities_col.isna() | (activities_col.str.strip() == '') | (activities_col.str.lower() == 'nan')
    empty_count = empty_activities.sum()
    
    print(f"      Empty/missing activities: {empty_count}/{len(df)}")
    
    if empty_count < len(df):
        # Sample some activities
        valid_activities = activities_col[~empty_activities].head(3)
        print(f"      Sample activities:")
        for activity in valid_activities:
            print(f"         '{activity}'")
        
        # Check for comma-separated activities (multiple per requirement)
        comma_separated = activities_col.str.contains(',', na=False).sum()
        print(f"      Requirements with multiple activities: {comma_separated}")

def check_compatibility(matcher_df, gt_df):
    """Check compatibility between matcher results and ground truth."""
    print(f"\n🔗 COMPATIBILITY CHECK")
    print("-" * 30)
    
    # Get requirement IDs from both sources
    matcher_req_col = None
    for col in matcher_df.columns:
        if 'id' in col.lower() and ('requirement' in col.lower() or col.lower() == 'id'):
            matcher_req_col = col
            break
    
    if not matcher_req_col:
        matcher_req_col = matcher_df.columns[0]
    
    gt_req_col = None
    for col in gt_df.columns:
        if col.lower() in ['id', 'requirement_id', 'req_id']:
            gt_req_col = col
            break
    
    if not gt_req_col:
        gt_req_col = gt_df.columns[0]
    
    matcher_ids = set(matcher_df[matcher_req_col].astype(str).str.strip())
    gt_ids = set(gt_df[gt_req_col].astype(str).str.strip())
    
    print(f"   Matcher requirement IDs: {len(matcher_ids)}")
    print(f"   Ground truth requirement IDs: {len(gt_ids)}")
    
    overlap = matcher_ids & gt_ids
    only_matcher = matcher_ids - gt_ids
    only_gt = gt_ids - matcher_ids
    
    print(f"   Overlapping IDs: {len(overlap)}")
    print(f"   Only in matcher: {len(only_matcher)}")
    print(f"   Only in ground truth: {len(only_gt)}")
    
    if len(overlap) == 0:
        print(f"   ❌ CRITICAL: No overlapping requirement IDs!")
        print(f"   Sample matcher IDs: {list(matcher_ids)[:5]}")
        print(f"   Sample GT IDs: {list(gt_ids)[:5]}")
    elif len(overlap) < min(len(matcher_ids), len(gt_ids)) * 0.5:
        print(f"   ⚠️  WARNING: Low overlap rate ({len(overlap)/min(len(matcher_ids), len(gt_ids)):.1%})")
    else:
        print(f"   ✅ Good overlap rate ({len(overlap)/min(len(matcher_ids), len(gt_ids)):.1%})")
    
    return len(overlap) > 0

def normalize_activity_test():
    """Test activity normalization function."""
    print(f"\n🧪 NORMALIZATION TEST")
    print("-" * 30)
    
    test_cases = [
        "2.3.2 Regulate System Temperature(context Thermal Control CSC)",
        "1.1.7 Monitor Fiber Quality(context Spectroscopy Sensor)",
        "2.2.3 Distribute Power(context Power Distribution)",
        "Regulate System Temperature",
        "Monitor Thermal Environment"
    ]
    
    def normalize_text(text):
        if pd.isna(text) or text == '':
            return ""
        text = str(text).strip()
        # Remove leading numbers
        text = re.sub(r'^\d+(\.\d+)*\s+', '', text)
        # Remove context information
        text = text.split('(context')[0]
        # Clean up spacing
        text = re.sub(r'\s+', ' ', text)
        return text.strip().lower().replace("-", " ")
    
    print("   Testing normalization function:")
    for test_case in test_cases:
        normalized = normalize_text(test_case)
        print(f"   '{test_case}' → '{normalized}'")

def main():
    """Run data verification."""
    print("🚀 STARTING DATA VERIFICATION")
    
    # Verify files
    files_ok = verify_data_files()
    
    # Test normalization
    normalize_activity_test()
    
    print(f"\n📋 NEXT STEPS")
    print("-" * 20)
    
    if files_ok:
        print("✅ Data files look good - ready to run fixed evaluation!")
        print("   Run: python fixed_simple_evaluation.py")
    else:
        print("❌ Fix data file issues before proceeding")
        print("   Check file paths and fix missing files")
    
    print("\n🔧 If issues persist:")
    print("   1. Check file encodings (try UTF-8, latin-1)")
    print("   2. Verify column names match expected format")
    print("   3. Ensure requirement IDs are consistent")

if __name__ == "__main__":
    main()