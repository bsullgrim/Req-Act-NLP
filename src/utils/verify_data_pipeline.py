# Add this verification script to test your data pipeline
# Save as verify_data_pipeline.py and run it

import pandas as pd
import sys
from pathlib import Path

def verify_data_pipeline():
    """Verify that the data pipeline is working correctly."""
    
    print("🔍 VERIFYING DATA PIPELINE")
    print("=" * 50)
    
    # 1. Check if input files exist
    files_to_check = [
        'requirements.csv',
        'activities.csv', 
        'manual_matches.csv'
    ]
    
    for file in files_to_check:
        if Path(file).exists():
            df = pd.read_csv(file)
            print(f"✅ {file}: {df.shape[0]} rows, {df.shape[1]} columns")
            print(f"   Columns: {list(df.columns)}")
        else:
            print(f"❌ {file}: NOT FOUND")
    
    # 2. Check output directory structure
    output_dirs = [
        'outputs/matching_results',
        'outputs/evaluation_results/dashboards',
        'outputs/quality_analysis'
    ]
    
    print(f"\n📁 OUTPUT DIRECTORIES:")
    for dir_path in output_dirs:
        path = Path(dir_path)
        if path.exists():
            files = list(path.glob('*'))
            print(f"✅ {dir_path}: {len(files)} files")
            for file in files[:3]:  # Show first 3 files
                print(f"   - {file.name}")
            if len(files) > 3:
                print(f"   ... and {len(files)-3} more")
        else:
            print(f"❌ {dir_path}: NOT FOUND")
    
    # 3. Check latest matching results
    matching_results = Path('outputs/matching_results')
    if matching_results.exists():
        csv_files = list(matching_results.glob('*.csv'))
        if csv_files:
            latest_file = max(csv_files, key=lambda x: x.stat().st_mtime)
            df = pd.read_csv(latest_file)
            print(f"\n📊 LATEST MATCHING RESULTS: {latest_file.name}")
            print(f"   - Shape: {df.shape}")
            print(f"   - Columns: {list(df.columns)}")
            print(f"   - Unique requirements: {df['ID'].nunique() if 'ID' in df.columns else 'Unknown'}")
            print(f"   - Score range: {df['Combined Score'].min():.3f} - {df['Combined Score'].max():.3f}" if 'Combined Score' in df.columns else "No scores")
            
            # Sample data
            print(f"   - Sample rows:")
            for i, (_, row) in enumerate(df.head(3).iterrows()):
                req_id = row.get('ID', 'N/A')
                activity = str(row.get('Activity Name', 'N/A'))[:40] + "..."
                score = row.get('Combined Score', 0)
                print(f"     {i+1}. {req_id}: {activity} (score: {score:.3f})")
        else:
            print(f"\n❌ No CSV files found in matching results")
    
    # 4. Check dashboard file
    dashboard_file = Path('outputs/evaluation_results/dashboards/unified_evaluation_dashboard.html')
    if dashboard_file.exists():
        file_size = dashboard_file.stat().st_size / 1024  # KB
        print(f"\n🌐 DASHBOARD FILE:")
        print(f"   - Size: {file_size:.1f} KB")
        print(f"   - Created: {dashboard_file.stat().st_mtime}")
        
        # Check if it contains predictions table
        with open(dashboard_file, 'r', encoding='utf-8') as f:
            content = f.read()
            
        table_indicators = [
            'id="predictions-table"',
            'class="data-row"',
            'data-req-id=',
            'showRowDetails'
        ]
        
        print(f"   - Content checks:")
        for indicator in table_indicators:
            count = content.count(indicator)
            status = "✅" if count > 0 else "❌"
            print(f"     {status} {indicator}: {count} occurrences")
            
        # Count data rows
        data_row_count = content.count('class="data-row"')
        print(f"   - Estimated data rows in HTML: {data_row_count}")
        
    else:
        print(f"\n❌ Dashboard file not found")
    
    print(f"\n🎯 SUMMARY:")
    print(f"   - If 'Estimated data rows in HTML' is 1, the problem is in data processing")
    print(f"   - If predictions CSV has many rows but HTML has 1, the problem is in table generation")
    print(f"   - Check the console output from the debug code for more details")

if __name__ == "__main__":
    verify_data_pipeline()