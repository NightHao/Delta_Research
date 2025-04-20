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
        '/home/yeskey525/iso_data/ISO_15118-3_chunk.pkl',
        '/home/yeskey525/iso_data/iso5_chunk.pkl',
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
        '/home/yeskey525/Research_CODE/experiments/golden_answers/visualizations',
        '/home/yeskey525/Research_CODE/experiments/golden_answers/prompts',
        '/home/yeskey525/Research_CODE/experiments/golden_answers/tree_json'
    ]
    
    for d in dirs:
        if not os.path.exists(d):
            os.makedirs(d)
            print(f"Created directory: {d}")

def main():
    """Run the complete analysis pipeline"""
    print("Starting prompt generation...")
    
    # Use the Lewis environment Python
    python_interpreter = "/home/yeskey525/anaconda3/envs/Lewis/bin/python"
    
    # Get the current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Generate prompts from gathered information
    prompt_script = os.path.join(current_dir, 'generate_answer_prompts.py')
    success = run_command(
        f"{python_interpreter} {prompt_script} --input gathered_information_with_images.json",
        "Generating Answer Prompts"
    )
    
    if not success:
        print("Prompt generation failed.")
        sys.exit(1)
    
    print("\n\nPrompt generation completed successfully!")
    print("Prompts available in:")
    print("  - /home/yeskey525/Research_CODE/experiments/golden_answers/prompts/")

if __name__ == "__main__":
    main() 