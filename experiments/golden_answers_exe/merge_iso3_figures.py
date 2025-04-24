#!/usr/bin/env python3
"""
Script to merge the content from iso3-images_ans.json into iso3_tree.json
matching image descriptions with figures in the ISO tree.
"""

import json
import os
import re
import sys

# Paths to the input files
ISO_TREE_PATH = '/home/yeskey525/Research_CODE/experiments/golden_answers/iso3_tree.json'
ISO_IMAGES_PATH = '/home/yeskey525/Research_CODE/experiments/golden_answers/iso3-images_ans.json'
OUTPUT_PATH = '/home/yeskey525/Research_CODE/experiments/golden_answers/iso3_tree_with_images.json'

def load_json(file_path):
    """Load and return JSON data from a file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        sys.exit(1)

def find_figure_node(tree, figure_pattern):
    """Recursively search the tree for a node matching the figure pattern."""
    # Check if current node matches the pattern
    if 'name' in tree and re.search(figure_pattern, tree['name'], re.IGNORECASE):
        return tree
    
    # Check children
    if 'children' in tree:
        for child in tree['children']:
            result = find_figure_node(child, figure_pattern)
            if result:
                return result
    
    return None

def find_matching_figure_patterns(image_key, images_content, tree_json):
    """
    Try to match the image with a figure in the tree based on patterns.
    Returns a list of potential matching patterns.
    """
    patterns = []
    
    # Extract figure numbers from image keys
    if image_key.startswith("iso3-image_"):
        num = image_key.split(".")[0].split("_")[1]
        patterns.append(f"Figure {num}")
    
    # Extract figure numbers from image content (description)
    img_content = images_content[image_key]
    figure_matches = re.findall(r"Figure\s+(\d+(?:\.\d+)?)", img_content, re.IGNORECASE)
    for fig_num in figure_matches:
        patterns.append(f"Figure {fig_num}")
    
    # Special handling for direct figure name matches
    for key in images_content:
        if key.startswith("Figure "):
            if image_key == key or re.search(r"Figure\s+" + key.split("Figure ")[1].split(" —")[0], img_content, re.IGNORECASE):
                patterns.append(key)
    
    return patterns

def merge_images_into_tree(tree_json, images_content):
    """Merge image descriptions into the ISO tree."""
    # Create a dictionary to track processed images
    processed_images = {}
    
    # First, handle direct figure name matches
    direct_matches = {}
    for key in images_content:
        if key.startswith("Figure "):
            # Try to find the node with this exact name
            node = find_figure_node(tree_json, re.escape(key))
            if node:
                if 'image_description' not in node:
                    node['image_description'] = images_content[key]
                    direct_matches[key] = True
    
    # Now handle the image filename matches
    for image_key in images_content:
        # Skip already processed direct figure matches
        if image_key in direct_matches:
            processed_images[image_key] = True
            continue
        
        if not image_key.startswith("iso3-image_"):
            continue
        
        patterns = find_matching_figure_patterns(image_key, images_content, tree_json)
        
        for pattern in patterns:
            node = find_figure_node(tree_json, re.escape(pattern))
            if node:
                if 'image_description' not in node:
                    node['image_description'] = images_content[image_key]
                    processed_images[image_key] = True
                    break
    
    # Report unmatched images
    unmatched = [key for key in images_content if key not in processed_images]
    if unmatched:
        print(f"Warning: Could not match {len(unmatched)} images:")
        for key in unmatched:
            print(f"  - {key}")
    
    return tree_json

def main():
    # Load JSON data
    tree_json = load_json(ISO_TREE_PATH)
    images_content = load_json(ISO_IMAGES_PATH)
    
    # Merge the content
    merged_tree = merge_images_into_tree(tree_json, images_content)
    
    # Save the merged result
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(merged_tree, f, ensure_ascii=False, indent=2)
    
    print(f"Merged tree saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main() 