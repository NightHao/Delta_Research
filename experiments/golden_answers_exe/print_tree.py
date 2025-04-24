#!/usr/bin/env python3
"""
Print the tree structure from pickle files
"""
import os
import sys
import pickle
import json
from anytree import Node, RenderTree
import argparse

def load_pickle_file(pickle_path):
    """Load content from a pickle file"""
    try:
        with open(pickle_path, 'rb') as f:
            content = pickle.load(f)
        print(f"Successfully loaded content from {pickle_path}")
        return content
    except Exception as e:
        print(f"Error loading content from {pickle_path}: {e}")
        return None

def build_tree(content, source_name=None):
    """
    Build a tree structure from the content chunks
    """
    root = Node("Root")  # Root node
    nodes = {}  # Store created nodes
    
    for item in content:
        current_parent = root
        for level in range(1, 8):  # iterate section_level_1 to section_level_7
            key = f"section_level_{level}"
            value = item.get(key, "").strip()
            if value:  # If current level has a value
                if value not in nodes:  # If this node hasn't been created yet
                    section_id = " > ".join(
                        [item.get(f"section_level_{lvl}", "").strip() for lvl in range(1, level + 1)]
                    )
                    nodes[value] = Node(value, parent=current_parent, section_id=section_id, source=source_name)
                current_parent = nodes[value]  # Update parent node
        
        # Add text content as child node
        text_value = item.get("text", "").strip()
        if text_value:
            section_id = " > ".join(
                [item.get(f"section_level_{lvl}", "").strip() for lvl in range(1, 8)]
            )
            Node(text_value, parent=current_parent, section_id=section_id, source=source_name)
    
    return root

def print_tree(tree, max_level=None, search_text=None):
    """Print the tree structure, optionally limiting depth and searching for text"""
    print(f"\nTree structure (total nodes: {len(tree.descendants) + 1}):")
    
    found_nodes = []
    
    for pre, _, node in RenderTree(tree):
        # Skip if we're over the max level
        if max_level is not None and node.depth > max_level:
            continue
        
        # Check if we're searching for specific text
        if search_text:
            if search_text.lower() in node.name.lower():
                found_nodes.append(node)
                print(f"{pre}{node.name} [MATCH]")
            # Skip non-matching nodes when searching
            continue
        
        # Print the node
        print(f"{pre}{node.name}")
    
    # If we were searching and found results, print a summary
    if search_text and found_nodes:
        print(f"\nFound {len(found_nodes)} nodes containing '{search_text}':")
        for node in found_nodes:
            print(f"- {node.name}")
            # Print parent path
            path = []
            current = node.parent
            while current:
                path.append(current.name)
                current = current.parent
            path.reverse()
            print(f"  Path: {' > '.join(path)}")
    elif search_text:
        print(f"\nNo nodes found containing '{search_text}'")

def tree_to_dict(node):
    """Convert a tree node to a dictionary for JSON serialization"""
    result = {
        'name': node.name,
        'path': '/'.join([n.name for n in node.path]),
        'depth': node.depth
    }
    
    # Add other attributes
    for key, value in node.__dict__.items():
        if key not in ['name', 'parent', 'children'] and not key.startswith('_'):
            result[key] = value
    
    # Add children recursively
    if node.children:
        result['children'] = [tree_to_dict(child) for child in node.children]
    
    return result

def save_tree_to_json(tree, output_file):
    """Save the tree structure to a JSON file"""
    tree_dict = tree_to_dict(tree)
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(tree_dict, f, indent=2, ensure_ascii=False)
        print(f"Tree saved to {output_file}")
        return True
    except Exception as e:
        print(f"Error saving tree to JSON: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Print tree structure from pickle files')
    parser.add_argument('--file', '-f', type=str, help='Path to pickle file to load', 
                      default='/home/yeskey525/Research_CODE/experiments/golden_answers/ISO_15118-3_chunk.pkl')
    parser.add_argument('--max-level', '-m', type=int, help='Maximum depth level to print')
    parser.add_argument('--search', '-s', type=str, help='Search for text in node names')
    parser.add_argument('--output', '-o', type=str, help='Save tree to JSON file')
    args = parser.parse_args()
    
    # Check if the file exists
    if not os.path.exists(args.file):
        print(f"Error: File {args.file} does not exist")
        return
    
    # Load content
    content = load_pickle_file(args.file)
    if not content:
        print("Failed to load content. Exiting.")
        return
    
    # Build the tree
    tree = build_tree(content, os.path.basename(args.file))
    if not tree:
        print("Failed to build tree. Exiting.")
        return
    
    # Print the tree
    print_tree(tree, args.max_level, args.search)
    
    # Save to JSON if requested
    if args.output:
        save_tree_to_json(tree, args.output)

if __name__ == "__main__":
    main() 