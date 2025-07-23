"""
Run All Metrics Generation
==========================
This script runs all metrics generation scripts in the correct order
to provide comprehensive evaluation of the MIG Weld Defect Detection system.

Author: Debraj Mukherjee
Project: MIG Weld Defect Detection
"""

import os
import sys
import subprocess
import time
from pathlib import Path

def run_script(script_path, script_name):
    """Run a Python script and handle errors"""
    print(f"\n{'='*60}")
    print(f"🚀 Running: {script_name}")
    print(f"{'='*60}")
    
    try:
        start_time = time.time()
        result = subprocess.run([sys.executable, script_path], 
                              capture_output=True, text=True, cwd=os.getcwd())
        end_time = time.time()
        
        if result.returncode == 0:
            print(f"✅ {script_name} completed successfully!")
            print(f"⏱️ Execution time: {end_time - start_time:.2f} seconds")
            if result.stdout:
                print("📄 Output:")
                print(result.stdout[-500:])  # Show last 500 characters
        else:
            print(f"❌ {script_name} failed with error:")
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ Error running {script_name}: {e}")
        return False
    
    return True

def main():
    """Main execution function"""
    print("🎯 MIG Weld Defect Detection - Comprehensive Metrics Generation")
    print("="*70)
    
    # Get base directory
    base_dir = Path(__file__).parent.parent
    scripts_dir = base_dir / "scripts"
    
    # List of scripts to run in order
    scripts_to_run = [
        {
            'path': scripts_dir / "enhanced_metrics_visuals.py",
            'name': "Enhanced Metrics Visuals",
            'description': "Generate ROC curves, precision-recall curves, and enhanced confusion matrix"
        },
        {
            'path': scripts_dir / "comprehensive_classifier_metrics.py", 
            'name': "Comprehensive Classifier Metrics",
            'description': "Generate detailed classifier performance analysis"
        },
        {
            'path': scripts_dir / "evaluate_yolo_weights.py",
            'name': "YOLO Weights Evaluation", 
            'description': "Evaluate performance of all YOLO model weights"
        },
        {
            'path': scripts_dir / "generate_metrics_visuals.py",
            'name': "Additional Metrics Visuals",
            'description': "Generate additional performance visualizations"
        }
    ]
    
    print("📋 Scripts to execute:")
    for i, script in enumerate(scripts_to_run, 1):
        print(f"   {i}. {script['name']}: {script['description']}")
    
    print(f"\n📁 Results will be saved to: {base_dir / 'results'}")
    
    # Confirm execution
    user_input = input("\n🤔 Do you want to proceed? (y/N): ").strip().lower()
    if user_input not in ['y', 'yes']:
        print("❌ Execution cancelled by user.")
        return
    
    # Execute scripts
    successful_runs = 0
    failed_runs = 0
    
    start_total_time = time.time()
    
    for script in scripts_to_run:
        if not script['path'].exists():
            print(f"⚠️ Script not found: {script['path']}")
            failed_runs += 1
            continue
        
        success = run_script(str(script['path']), script['name'])
        if success:
            successful_runs += 1
        else:
            failed_runs += 1
            
            # Ask user if they want to continue after a failure
            continue_input = input(f"\n❓ Continue with remaining scripts? (Y/n): ").strip().lower()
            if continue_input in ['n', 'no']:
                print("🛑 Execution stopped by user.")
                break
    
    end_total_time = time.time()
    
    # Summary
    print("\n" + "="*70)
    print("🎉 METRICS GENERATION COMPLETE!")
    print("="*70)
    print(f"✅ Successful runs: {successful_runs}")
    print(f"❌ Failed runs: {failed_runs}")
    print(f"⏱️ Total execution time: {end_total_time - start_total_time:.2f} seconds")
    
    if successful_runs > 0:
        results_dir = base_dir / "results"
        print(f"\n📁 Generated files in {results_dir}:")
        
        # List generated files
        if results_dir.exists():
            result_files = list(results_dir.glob("*.png")) + list(results_dir.glob("*.csv")) + list(results_dir.glob("*.txt"))
            for file in sorted(result_files):
                file_size = file.stat().st_size / 1024  # KB
                print(f"   📄 {file.name} ({file_size:.1f} KB)")
        
        print("\n💡 Next Steps:")
        print("   • Review generated visualizations and reports")
        print("   • Update README with new performance metrics")
        print("   • Consider model optimization based on results")
        print("   • Share results with stakeholders")
    
    if failed_runs > 0:
        print(f"\n⚠️ {failed_runs} script(s) failed. Check error messages above.")
        print("💡 Troubleshooting tips:")
        print("   • Ensure all dependencies are installed: pip install -r requirements.txt")
        print("   • Check that model files exist in the weights/ directory")
        print("   • Verify that defect_features_balanced.csv exists")
        print("   • Run scripts individually to debug specific issues")

if __name__ == "__main__":
    main()
