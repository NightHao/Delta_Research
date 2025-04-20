#!/usr/bin/env python3
"""
Script to update all prompts by adding image descriptions from the iso3_tree_with_images.json file
when figures are referenced in the prompts.
"""

import os
import json
import re
import glob

# Paths
PROMPTS_DIR = '/home/yeskey525/Research_CODE/experiments/golden_answers/complete_prompts'
JSON_TREE_PATH = '/home/yeskey525/Research_CODE/experiments/golden_answers/iso3_tree_with_images.json'
GATHERED_INFO_PATH = '/home/yeskey525/Research_CODE/experiments/golden_answers/gathered_information_with_images.json'

def load_json_file(file_path):
    """Load and return JSON data from a file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def extract_image_descriptions_from_tree(tree_data):
    """Extract all image descriptions from the JSON tree"""
    image_descriptions = {}
    
    def process_node(node):
        if 'name' in node and 'image_description' in node and node['image_description']:
            # Store by both name and figure reference
            image_descriptions[node['name']] = node['image_description']
            
            # Check if it's a figure and extract figure number
            figure_match = re.search(r'Figure\s+([A-Z0-9\.]+)', node['name'], re.IGNORECASE)
            if figure_match:
                figure_ref = figure_match.group(0)  # "Figure X.Y"
                image_descriptions[figure_ref] = node['image_description']
        
        if 'children' in node and node['children']:
            for child in node['children']:
                process_node(child)
    
    process_node(tree_data)
    return image_descriptions

def extract_image_descriptions_from_gathered(gathered_data):
    """Extract image descriptions from gathered information JSON"""
    image_descriptions = {}
    
    for question in gathered_data:
        for info in question.get('gathered_information', []):
            if 'image_description' in info and info['image_description']:
                # If there's a reference to a figure in the text
                figure_match = re.search(r'Figure\s+([A-Z0-9\.]+)', info['text'], re.IGNORECASE)
                if figure_match:
                    figure_ref = figure_match.group(0)  # "Figure X.Y"
                    image_descriptions[figure_ref] = info['image_description']
                    
                # Also store by text to increase matching chances
                image_descriptions[info['text'][:50]] = info['image_description']
    
    return image_descriptions

def update_prompts(prompts_dir, image_descriptions):
    """Update all prompts in the directory by adding image descriptions"""
    updated_count = 0
    prompts = glob.glob(os.path.join(prompts_dir, "*.txt"))
    
    print(f"Found {len(prompts)} prompt files to process")
    
    for prompt_path in prompts:
        with open(prompt_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check for figure references in the content
        figure_references = re.findall(r'Figure\s+([A-Z0-9\.]+)', content, re.IGNORECASE)
        
        updated = False
        
        # For each figure reference found
        for fig_ref in figure_references:
            full_ref = f"Figure {fig_ref}"
            
            # Check if we have an image description for this figure
            if full_ref in image_descriptions:
                # Check if the image description is already in the prompt
                if image_descriptions[full_ref] not in content:
                    # Find the position of the reference in the content
                    ref_pos = content.find(full_ref)
                    if ref_pos != -1:
                        # Find the end of the line containing the reference
                        line_end = content.find('\n', ref_pos)
                        if line_end == -1:  # If it's the last line
                            line_end = len(content)
                        
                        # Insert the image description after the line containing the reference
                        desc_text = f"\n\nImage Description for {full_ref}:\n{image_descriptions[full_ref]}\n"
                        content = content[:line_end] + desc_text + content[line_end:]
                        updated = True
                        print(f"Added image description for {full_ref} in {os.path.basename(prompt_path)}")
        
        # Save the updated content back to the file if changes were made
        if updated:
            with open(prompt_path, 'w', encoding='utf-8') as f:
                f.write(content)
            updated_count += 1
    
    return updated_count

def main():
    # Load the JSON tree
    tree_data = load_json_file(JSON_TREE_PATH)
    if not tree_data:
        print(f"Failed to load tree data from {JSON_TREE_PATH}")
        return
    
    # Extract image descriptions from the tree
    tree_image_descriptions = extract_image_descriptions_from_tree(tree_data)
    print(f"Extracted {len(tree_image_descriptions)} image descriptions from the JSON tree")
    
    # Load the gathered information
    gathered_data = load_json_file(GATHERED_INFO_PATH)
    if gathered_data:
        # Extract image descriptions from gathered information
        gathered_image_descriptions = extract_image_descriptions_from_gathered(gathered_data)
        print(f"Extracted {len(gathered_image_descriptions)} image descriptions from gathered information")
        
        # Merge the two sets of image descriptions, preferring tree descriptions if there's a conflict
        image_descriptions = {**gathered_image_descriptions, **tree_image_descriptions}
    else:
        print(f"No gathered information found at {GATHERED_INFO_PATH}, using only tree data")
        image_descriptions = tree_image_descriptions
    
    # Update all prompts with image descriptions
    updated_count = update_prompts(PROMPTS_DIR, image_descriptions)
    print(f"Updated {updated_count} prompt files with image descriptions")

if __name__ == "__main__":
    main() 