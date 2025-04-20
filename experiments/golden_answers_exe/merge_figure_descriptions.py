#!/usr/bin/env python3
"""
Merge figure descriptions into the JSON tree
"""
import os
import json
import argparse
import re
from pathlib import Path

def load_json_file(file_path):
    """Load data from a JSON file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"Successfully loaded data from {file_path}")
        return data
    except Exception as e:
        print(f"Error loading JSON file {file_path}: {e}")
        return None

def save_json_file(data, file_path):
    """Save data to a JSON file"""
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Successfully saved data to {file_path}")
        return True
    except Exception as e:
        print(f"Error saving JSON file {file_path}: {e}")
        return False

def load_figure_descriptions(file_path):
    """Load figure descriptions from a separate JSON file"""
    figures = load_json_file(file_path)
    if not figures:
        print("No figure descriptions found")
        return {}
    
    # Check if the figure data is in string format rather than dict format
    if isinstance(figures, dict) and any(isinstance(val, str) for val in figures.values()):
        print("Converting string descriptions to structured format...")
        structured_figures = {}
        for figure_id, description in figures.items():
            structured_figures[figure_id] = {
                'id': figure_id,
                'text': description,
                'description': description,
                'source': 'iso3-images'
            }
        figures = structured_figures
    
    print(f"Loaded {len(figures)} figure descriptions")
    return figures

def find_node_by_id(tree, node_id):
    """Find a node by its ID in the tree"""
    if tree.get('id') == node_id:
        return tree
    
    children = tree.get('children', [])
    for child in children:
        result = find_node_by_id(child, node_id)
        if result:
            return result
    
    return None

def find_nodes_by_text_pattern(tree, pattern, results=None):
    """Find nodes by a text pattern in the tree"""
    if results is None:
        results = []
    
    # Check current node
    node_text = tree.get('text', '')
    if pattern.search(node_text):
        results.append(tree)
    
    # Check children recursively
    children = tree.get('children', [])
    for child in children:
        find_nodes_by_text_pattern(child, pattern, results)
    
    return results

def merge_figures_into_tree(tree, figures):
    """Merge figure descriptions into the tree"""
    if not tree or not figures:
        print("Cannot merge: tree or figures missing")
        return tree
    
    figure_nodes_added = 0
    
    # For each figure, find its appropriate parent in the tree or add it as a new node
    for figure_id, figure_data in figures.items():
        # Make sure figure_data is a dictionary
        if isinstance(figure_data, str):
            figure_data = {
                'id': figure_id,
                'text': figure_data,
                'description': figure_data,
                'source': 'iso3-images'
            }
        
        parent_id = figure_data.get('parent_id', None)
        figure_text = figure_data.get('text', '')
        figure_caption = figure_data.get('caption', '')
        figure_description = figure_data.get('description', '')
        
        # Skip if no text or description
        if not figure_text and not figure_description:
            continue
        
        # Create the figure node
        figure_node = {
            'id': figure_id,
            'text': figure_caption or figure_text,
            'node_type': 'figure',
            'metadata': {
                'description': figure_description,
                'source': figure_data.get('source', 'unknown'),
                'page': figure_data.get('page', None)
            }
        }
        
        # Try to find a parent by ID if provided
        parent_node = None
        if parent_id:
            parent_node = find_node_by_id(tree, parent_id)
        
        # If parent not found by ID, try to find it by text pattern
        if not parent_node and figure_text:
            try:
                # Create a pattern that looks for the figure number
                pattern = re.compile(r'Figure\s+\d+', re.IGNORECASE)
                candidate_nodes = find_nodes_by_text_pattern(tree, pattern)
                
                # Find the best match
                for node in candidate_nodes:
                    if figure_text.lower() in node.get('text', '').lower():
                        parent_node = node
                        break
            except Exception as e:
                print(f"Error finding parent for figure {figure_id}: {e}")
        
        # Add the figure node to the tree
        if parent_node:
            # Add as a child of the found parent
            if 'children' not in parent_node:
                parent_node['children'] = []
            parent_node['children'].append(figure_node)
            figure_node['parent_id'] = parent_node.get('id', '')
            figure_nodes_added += 1
        else:
            # If no parent found, add to the root level
            print(f"No parent found for figure {figure_id}, adding to root")
            if 'children' not in tree:
                tree['children'] = []
            tree['children'].append(figure_node)
            figure_nodes_added += 1
    
    print(f"Added {figure_nodes_added} figure nodes to the tree")
    return tree

def main():
    parser = argparse.ArgumentParser(description="Merge figure descriptions into the JSON tree")
    parser.add_argument('--tree', type=str, 
                        default='/home/yeskey525/Research_CODE/experiments/golden_answers/tree_json/combined_tree.json',
                        help='Path to the JSON tree file')
    parser.add_argument('--figures', type=str, 
                        default='/home/yeskey525/Research_CODE/experiments/golden_answers/iso3-images_ans.json',
                        help='Path to the figure descriptions JSON file')
    parser.add_argument('--output', type=str, 
                        default='/home/yeskey525/Research_CODE/experiments/golden_answers/tree_json/combined_tree_with_figures.json',
                        help='Path to save the merged tree')
    args = parser.parse_args()
    
    # Load the tree
    tree = load_json_file(args.tree)
    if not tree:
        print("Failed to load tree. Exiting.")
        return False
    
    # Check if figures file exists, if not, create a dummy one
    if not os.path.exists(args.figures):
        print(f"Warning: Figure descriptions file {args.figures} not found.")
        print("Creating an empty figure descriptions file...")
        dummy_figures = {}
        save_json_file(dummy_figures, args.figures)
    
    # Load figure descriptions
    figures = load_figure_descriptions(args.figures)
    
    # Merge figures into the tree
    merged_tree = merge_figures_into_tree(tree, figures)
    
    # Save the merged tree
    success = save_json_file(merged_tree, args.output)
    
    if success:
        print(f"Successfully merged figure descriptions into tree and saved to {args.output}")
        return True
    else:
        print("Failed to save merged tree")
        return False

if __name__ == "__main__":
    main() 