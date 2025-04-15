#!/usr/bin/env python3
"""
Run the complete information gathering and analysis pipeline.
"""

import os
import sys
import subprocess
import time

def run_command(command, description):
    """Run a shell command and display its output"""
    print(f"\n{'='*80}\n{description}\n{'='*80}")
    start_time = time.time()
    
    try:
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
                                   universal_newlines=True)
        
        # Display output in real-time
        for line in process.stdout:
            print(line.strip())
        
        process.wait()
        
        if process.returncode != 0:
            print(f"Error: Command failed with return code {process.returncode}")
            return False
        
    except Exception as e:
        print(f"Error executing command: {e}")
        return False
    
    elapsed_time = time.time() - start_time
    print(f"\nCompleted in {elapsed_time:.2f} seconds")
    return True

def check_files_exist():
    """Check if required files exist"""
    required_files = [
        '/home/yeskey525/Research_CODE/experiments/golden_answers/thesis_experiment_result_update.csv',
    ]
    
    # Both pickle files to be checked, but we'll proceed if at least one exists
    pickle_files = [
        '/home/yeskey525/Research_CODE/experiments/golden_answers/ISO_15118-3_chunk.pkl',
        '/home/yeskey525/Research_CODE/experiments/golden_answers/iso5_chunk.pkl',
        '/home/yeskey525/Research_CODE/experiments/golden_answers/create_chunks.pkl'
    ]
    
    # Check required files
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print("Error: The following required files are missing:")
        for f in missing_files:
            print(f"  - {f}")
        return False
    
    # Check pickle files
    available_pickle_files = [f for f in pickle_files if os.path.exists(f)]
    if not available_pickle_files:
        print("Error: None of the following pickle files exist:")
        for f in pickle_files:
            print(f"  - {f}")
        return False
    else:
        print("Found the following pickle files:")
        for f in available_pickle_files:
            print(f"  - {f}")
    
    return True

def create_directories():
    """Create any necessary directories"""
    dirs = [
        '/home/yeskey525/Research_CODE/experiments/golden_answers/visualizations'
    ]
    
    for d in dirs:
        if not os.path.exists(d):
            os.makedirs(d)
            print(f"Created directory: {d}")

def main():
    """Run the complete analysis pipeline"""
    print("Starting information gathering and analysis pipeline...")
    
    # Parse command line arguments
    debug_mode = "--debug" in sys.argv
    
    # Check if required files exist
    if not check_files_exist():
        sys.exit(1)
    
    # Create necessary directories
    create_directories()
    
    # Get the current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Step 1: Run the information gathering script with debug flag if specified
    gather_script = os.path.join(current_dir, 'gather_information.py')
    debug_flag = "--debug" if debug_mode else ""
    success = run_command(
        f"python {gather_script} {debug_flag}",
        "STEP 1: Gathering Information"
    )
    
    if not success:
        print("Information gathering failed. Stopping pipeline.")
        print("\nTrying again with the debug flag enabled...")
        success = run_command(
            f"python {gather_script} --debug",
            "STEP 1: Gathering Information (Debug Mode)"
        )
        if not success:
            print("Information gathering failed even in debug mode. Stopping pipeline.")
            sys.exit(1)
    
    # Step 2: Run the analysis script
    analyze_script = os.path.join(current_dir, 'analyze_gathered_info.py')
    success = run_command(
        f"python {analyze_script}",
        "STEP 2: Analyzing Gathered Information"
    )
    
    if not success:
        print("Analysis failed.")
        sys.exit(1)
    
    print("\n\nPipeline completed successfully!")
    print("Results available in:")
    print("  - /home/yeskey525/Research_CODE/experiments/golden_answers/gathered_information.json")
    print("  - /home/yeskey525/Research_CODE/experiments/golden_answers/analysis_report.md")
    print("  - /home/yeskey525/Research_CODE/experiments/golden_answers/visualizations/")

if __name__ == "__main__":
    main() 