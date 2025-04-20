#!/usr/bin/env python3
"""
Convert text answers to JSON format for golden answers
"""
import os
import json
import re
import argparse
from pathlib import Path

def ensure_directory(directory):
    """Ensure the directory exists"""
    os.makedirs(directory, exist_ok=True)
    return directory

def extract_question_from_filename(filename):
    """Extract the question from the filename"""
    # Remove the question number prefix if present
    match = re.match(r'(.+?)_(.+)\.txt', os.path.basename(filename))
    if match:
        return match.group(1)
    
    # Otherwise just return the filename without extension
    return os.path.splitext(os.path.basename(filename))[0]

def convert_answers_to_json(answers_dir, output_file):
    """Convert all text answers to a single JSON file"""
    answers_dict = {}
    
    # Get all txt files in the answers directory
    txt_files = [f for f in os.listdir(answers_dir) if f.endswith('.txt')]
    
    if not txt_files:
        print(f"No text files found in {answers_dir}")
        return False
    
    for filename in txt_files:
        file_path = os.path.join(answers_dir, filename)
        question = extract_question_from_filename(filename)
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                answer_text = f.read().strip()
                
                # Store the answer in the dictionary
                answers_dict[question] = answer_text
                print(f"Processed answer for question: {question}")
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    # Save to JSON
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(answers_dict, f, indent=2, ensure_ascii=False)
        print(f"Answers saved to {output_file}")
        return True
    except Exception as e:
        print(f"Error saving to JSON: {e}")
        return False

def process_direct_input(text_input, output_file):
    """Process a direct text input and convert it to JSON"""
    # Try to extract the question from the first line
    lines = text_input.strip().split('\n')
    
    # Look for a question pattern like "## Question: What is X?"
    question_match = re.search(r'(?:##\s*)?(?:Question:)?\s*(.+?)\?', lines[0])
    if question_match:
        question = question_match.group(1).strip() + "?"
    else:
        # Use a default question if not found
        question = "Unknown Question"
        print(f"Warning: Could not extract question from input. Using '{question}'")
    
    # Create a dictionary with the question and answer
    answers_dict = {question: text_input.strip()}
    
    # Save to JSON
    try:
        # If the output file already exists, load and update it
        if os.path.exists(output_file):
            with open(output_file, 'r', encoding='utf-8') as f:
                existing_answers = json.load(f)
            existing_answers.update(answers_dict)
            answers_dict = existing_answers
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(answers_dict, f, indent=2, ensure_ascii=False)
        print(f"Answer for '{question}' saved to {output_file}")
        return True
    except Exception as e:
        print(f"Error saving to JSON: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Convert answer texts to JSON format")
    parser.add_argument('--answers-dir', '-a', type=str, default='answers',
                        help='Directory containing text answer files')
    parser.add_argument('--output', '-o', type=str, default='golden_answers.json',
                        help='Output JSON file')
    parser.add_argument('--input-text', '-t', type=str, 
                        help='Direct text input instead of reading from files')
    parser.add_argument('--question', '-q', type=str,
                        help='Question for the direct text input')
    args = parser.parse_args()
    
    # Create the output directory if needed
    output_dir = os.path.join('/home/yeskey525/Research_CODE/experiments/golden_answers/ans')
    ensure_directory(output_dir)
    output_file = os.path.join(output_dir, args.output)
    
    if args.input_text:
        # Process direct text input
        text_input = args.input_text
        
        # If a specific question is provided, prepend it to the text
        if args.question:
            if not args.question.strip().endswith('?'):
                args.question += '?'
            text_input = f"## Question: {args.question}\n\n{text_input}"
        
        process_direct_input(text_input, output_file)
    else:
        # Process files in the answers directory
        answers_dir = os.path.join('/home/yeskey525/Research_CODE/experiments/golden_answers', args.answers_dir)
        if not os.path.exists(answers_dir):
            print(f"Answers directory {answers_dir} does not exist.")
            return
        
        convert_answers_to_json(answers_dir, output_file)

if __name__ == "__main__":
    main() 