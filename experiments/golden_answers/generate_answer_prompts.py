#!/usr/bin/env python3
"""
Generate answer prompts from gathered information
"""
import json
import os
import re
import argparse
from pathlib import Path

def load_gathered_info(json_file):
    """Load the gathered information from the JSON file"""
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"Successfully loaded data from {json_file}")
        return data
    except Exception as e:
        print(f"Error loading JSON file: {e}")
        return None

def clean_text(text):
    """Clean up text by removing excessive whitespace, etc."""
    if not text:
        return ""
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)
    # Remove leading/trailing whitespace
    text = text.strip()
    return text

def create_prompt_for_question(question_data):
    """Create a prompt for a single question based on its gathered information"""
    question_num = question_data.get('question_number', 'Unknown')
    question_text = question_data.get('question', 'Unknown')
    
    # Collect all information
    all_info = question_data.get('all_information', [])
    
    if not all_info:
        return f"Question {question_num}: {question_text}\n\nNo information available for this question."
    
    # Organize information by source
    table_info = [info for info in all_info if info.get('source') == 'table']
    figure_info = [info for info in all_info if info.get('source') == 'figure']
    answer_info = [info for info in all_info if info.get('source') == 'answer']
    
    # Build the prompt
    prompt = f"Question {question_num}: {question_text}\n\n"
    prompt += "Below is all the available information related to this question.\n"
    prompt += "Use ONLY this information to construct a comprehensive answer.\n"
    prompt += "Do not add any new information that is not present below.\n\n"
    
    # Add table information
    if table_info:
        prompt += "TABLE INFORMATION:\n"
        for i, info in enumerate(table_info, 1):
            text = clean_text(info.get('text', ''))
            parent = clean_text(info.get('parent_text', ''))
            if parent and parent != "No parent found":
                prompt += f"Table {i}: {parent} - {text}\n"
            else:
                prompt += f"Table {i}: {text}\n"
        prompt += "\n"
    
    # Add figure information
    if figure_info:
        prompt += "FIGURE INFORMATION:\n"
        for i, info in enumerate(figure_info, 1):
            text = clean_text(info.get('text', ''))
            parent = clean_text(info.get('parent_text', ''))
            if parent and parent != "No parent found":
                prompt += f"Figure {i}: {parent} - {text}\n"
            else:
                prompt += f"Figure {i}: {text}\n"
        prompt += "\n"
    
    # Add textual information
    if answer_info:
        prompt += "TEXTUAL INFORMATION:\n"
        for i, info in enumerate(answer_info, 1):
            text = clean_text(info.get('text', ''))
            parent = clean_text(info.get('parent_text', ''))
            
            # Include context hierarchy when available
            ancestors = info.get('ancestors', [])
            context = ""
            if ancestors:
                filtered_ancestors = [a for a in ancestors if a != "Root"]
                if filtered_ancestors:
                    context = " (Context: " + " > ".join(filtered_ancestors) + ")"
            
            if parent and parent != "No parent found":
                prompt += f"Text {i}: {parent}{context} - {text}\n"
            else:
                prompt += f"Text {i}: {text}\n"
        prompt += "\n"
        
        # Add siblings information as additional context
        prompt += "ADDITIONAL CONTEXT:\n"
        siblings_added = set()  # To avoid duplicates
        ancestors_added = set()  # To avoid duplicates
        
        # Process siblings and ancestors for all information types
        for info_type, type_name in [
            (answer_info, "Text"), 
            (table_info, "Table"), 
            (figure_info, "Figure")
        ]:
            for i, info in enumerate(info_type, 1):
                # Add siblings information
                siblings = info.get('siblings', [])
                if siblings:
                    for sibling in siblings:
                        sibling_text = clean_text(sibling)
                        if sibling_text and sibling_text not in siblings_added:
                            siblings_added.add(sibling_text)
                            prompt += f"Related Information {len(siblings_added)}: {sibling_text} (sibling of {type_name} {i})\n"
                
                # Add ancestor information (beyond just path)
                ancestors = info.get('ancestors', [])
                if ancestors:
                    for ancestor in ancestors:
                        ancestor_text = clean_text(ancestor)
                        if ancestor_text and ancestor_text not in ancestors_added and ancestor_text != "Root":
                            ancestors_added.add(ancestor_text)
                            prompt += f"Contextual Information {len(ancestors_added)}: {ancestor_text} (ancestor of {type_name} {i})\n"
        prompt += "\n"
    
    # Add output format instructions
    prompt += "OUTPUT FORMAT INSTRUCTIONS:\n"
    prompt += "Your answer should follow these guidelines:\n"
    prompt += "1. Start with an 'Overview' section that provides a concise explanation of what this concept/topic is.\n"
    prompt += "2. For the related content, organize it into appropriate sections with descriptive headings based on the information available.\n"
    prompt += "   - Analyze the content and group related information logically.\n"
    prompt += "   - Choose section titles that best represent the information, such as 'Related Topics', 'Related Primitives', 'Configuration Requirements', etc.\n"
    prompt += "   - Let the content guide your choice of section titles rather than forcing a predefined structure.\n"
    prompt += "3. The structure should match the format used in technical documentation, with clear hierarchy and organization.\n"
    prompt += "4. Include any relevant figures, tables, or processes mentioned in the information.\n"
    prompt += "5. Be as detailed and comprehensive as possible while ONLY using the provided information.\n"
    prompt += "6. In writing your answer, strictly follow the content provided in the information, and do not add any new information or make assumptions.\n"
    
    return prompt

def generate_prompts(data, output_dir):
    """Generate prompts for all questions and save them to files"""
    os.makedirs(output_dir, exist_ok=True)
    
    questions = data.get('questions', [])
    print(f"Generating prompts for {len(questions)} questions...")
    
    for question_data in questions:
        question_num = question_data.get('question_number', 'Unknown')
        question_text = question_data.get('question', '')
        
        # Create a safe filename
        safe_question = re.sub(r'[^\w\s-]', '', question_text)
        safe_question = re.sub(r'\s+', '_', safe_question).lower()
        filename = f"{question_num}_{safe_question[:50]}.txt"
        
        # Generate the prompt
        prompt = create_prompt_for_question(question_data)
        
        # Save to file
        output_file = os.path.join(output_dir, filename)
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(prompt)
        
        print(f"Created prompt for Question {question_num}: {output_file}")
    
    print(f"All prompts generated in {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Generate answer prompts from gathered information")
    parser.add_argument('--input', '-i', type=str, default='/home/yeskey525/Research_CODE/experiments/golden_answers/gathered_information.json',
                        help='Path to the gathered_information.json file')
    parser.add_argument('--output-dir', '-o', type=str, default='/home/yeskey525/Research_CODE/experiments/golden_answers/prompts',
                        help='Directory to save the generated prompts')
    args = parser.parse_args()
    
    # Load the gathered information
    data = load_gathered_info(args.input)
    if not data:
        print("Failed to load gathered information. Exiting.")
        return
    
    # Generate the prompts
    generate_prompts(data, args.output_dir)

if __name__ == "__main__":
    main() 