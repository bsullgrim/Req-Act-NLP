#!/usr/bin/env python3
"""
Evaluation Fix Runner
====================

Orchestrates the complete debugging and fixing process:
1. Data verification
2. Quick debug analysis  
3. Fixed evaluation with proper paths

Run this single script to diagnose and fix all evaluation issues.
"""

import sys
import subprocess
from pathlib import Path
import traceback

def run_script(script_name, description):
    """Run a Python script and capture output."""
    print(f"\n{'='*60}")
    print(f"🚀 RUNNING: {description}")
    print(f"{'='*60}")
    
    try:
        # Run the script
        result = subprocess.run([sys.executable, script_name], 
                              capture_output=True, 
                              text=True, 
                              timeout=300)  # 5 minute timeout
        
        # Print output
        if result.stdout:
            print(result.stdout)
        
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        
        if result.returncode == 0:
            print(f"✅ {description} completed successfully!")
            return True
        else:
            print(f"❌ {description} failed with return code {result.returncode}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"⏰ {description} timed out after 5 minutes")
        return False
    except Exception as e:
        print(f"❌ Error running {description}: {e}")
        traceback.print_exc()
        return False

def main():
    """Run the complete evaluation fix process."""
    
    print("🔧 EVALUATION FIX ORCHESTRATOR")
    print("=" * 60)
    print("This will run three scripts in sequence to diagnose and fix")
    print("the evaluation issues with your aerospace matcher.")
    print()
    
    # Check if scripts exist
    scripts = [
        ("data_verification_script.py", "Data Verification"),
        ("quick_debug_evaluation.py", "Quick Debug Analysis"),
        ("simple_evaluation.py", "Fixed Evaluation")
    ]
    
    missing_scripts = []
    for script_file, _ in scripts:
        if not Path(script_file).exists():
            missing_scripts.append(script_file)
    
    if missing_scripts:
        print(f"❌ Missing script files: {missing_scripts}")
        print("Please save the provided scripts first.")
        return
    
    print("✅ All script files found. Starting execution...")
    
    # Track success of each step
    results = {}
    
    # Step 1: Data Verification
    print(f"\n📋 STEP 1/3: Data Verification")
    print("Will check file formats, paths, and data consistency")
    results['verification'] = run_script("data_verification_script.py", "Data Verification")
    
    # Step 2: Quick Debug  
    print(f"\n📋 STEP 2/3: Quick Debug Analysis")
    print("Will analyze P.9 specifically and show what's happening")
    results['debug'] = run_script("quick_debug_evaluation.py", "Quick Debug Analysis")
    
    # Step 3: Fixed Evaluation
    print(f"\n📋 STEP 3/3: Fixed Evaluation")  
    print("Will run the corrected evaluation with proper debugging")
    results['evaluation'] = run_script("simple_evaluation.py", "Fixed Evaluation")
    
    # Summary
    print(f"\n{'='*60}")
    print("📊 EXECUTION SUMMARY")
    print(f"{'='*60}")
    
    for step, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"{step.upper():15}: {status}")
    
    total_success = sum(results.values())
    print(f"\nOverall: {total_success}/{len(results)} steps completed successfully")
    
    if total_success == len(results):
        print(f"\n🎉 ALL STEPS COMPLETED SUCCESSFULLY!")
        print(f"Check the outputs/evaluation_results/ directory for:")
        print(f"   • fixed_simple_metrics.json")
        print(f"   • fixed_simple_evaluation_report.txt")
        print(f"\nYour F1@5 score should now show the true performance!")
    elif results.get('verification', False):
        print(f"\n⚠️  PARTIAL SUCCESS")
        print(f"Data verification passed, but later steps had issues.")
        print(f"Check the error output above for details.")
    else:
        print(f"\n❌ CRITICAL ISSUES FOUND")
        print(f"Data verification failed. Check file paths and data formats.")
        print(f"Common issues:")
        print(f"   • Files not in data/raw/ directory")
        print(f"   • Missing aerospace_matches.csv in outputs/matching_results/")
        print(f"   • Encoding issues with CSV files")

if __name__ == "__main__":
    main()
