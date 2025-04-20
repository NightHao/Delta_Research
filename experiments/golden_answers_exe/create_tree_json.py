#!/usr/bin/env python3
"""
Convert ISO-5 PKL file to JSON tree and merge with existing ISO-3 tree
"""
import os
import pickle
import json
import argparse
from pathlib import Path

def load_pickle_file(file_path):
    """Load data from a pickle file"""
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        print(f"Successfully loaded data from {file_path}")
        
        # Print the type of data loaded
        print(f"Loaded data type: {type(data)}")
        
        # If it's a list, print its length and the first item's type
        if isinstance(data, list):
            print(f"List length: {len(data)}")
            if len(data) > 0:
                print(f"First item type: {type(data[0])}")
        # If it's a dict, print its keys
        elif isinstance(data, dict):
            print(f"Dict keys: {list(data.keys())}")
        
        return data
    except Exception as e:
        print(f"Error loading pickle file {file_path}: {e}")
        return None

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

def convert_node_to_dict(node, include_children=True):
    """Convert a node object to a dictionary representation"""
    if node is None:
        print("WARNING: Node is None")
        return None
    
    # Print the type and attributes of the node for debugging
    print(f"Node type: {type(node)}")
    if hasattr(node, '__dict__'):
        print(f"Node attributes: {sorted(node.__dict__.keys())}")
    else:
        print("Node has no __dict__ attribute")
    
    # Basic node properties
    node_dict = {
        'text': node.text if hasattr(node, 'text') else "",
        'node_type': node.node_type if hasattr(node, 'node_type') else "unknown",
        'id': node.id if hasattr(node, 'id') else "",
    }
    
    # Print node_dict for debugging
    print(f"Created node_dict: {node_dict}")
    
    # Optional properties
    if hasattr(node, 'parent') and node.parent is not None:
        node_dict['parent_id'] = node.parent.id if hasattr(node.parent, 'id') else ""
    
    # Add metadata if it exists
    if hasattr(node, 'metadata') and node.metadata:
        node_dict['metadata'] = node.metadata
    
    # Add vectors if they exist
    if hasattr(node, 'embedding') and node.embedding is not None:
        node_dict['embedding'] = node.embedding.tolist() if hasattr(node.embedding, 'tolist') else node.embedding
    
    # Add children recursively if requested
    if include_children and hasattr(node, 'children') and node.children:
        print(f"This node has {len(node.children)} children")
        node_dict['children'] = [
            convert_node_to_dict(child, include_children=True) 
            for child in node.children
        ]
    else:
        print("This node has no children")
    
    return node_dict

def save_json_tree(tree_dict, output_file):
    """Save the tree dictionary to a JSON file"""
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(tree_dict, f, ensure_ascii=False, indent=2)
        print(f"Successfully saved tree to {output_file}")
        return True
    except Exception as e:
        print(f"Error saving tree to JSON: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Convert ISO-5 PKL file to JSON tree and merge with existing ISO-3 tree")
    parser.add_argument('--iso3-tree', type=str, 
                        default='/home/yeskey525/Research_CODE/experiments/golden_answers/iso3_tree_with_images.json',
                        help='Path to the existing ISO-3 tree with images JSON file')
    parser.add_argument('--iso5', type=str, 
                        default='/home/yeskey525/iso_data/iso5_chunk.pkl',
                        help='Path to the iso5_chunk.pkl file')
    parser.add_argument('--output-dir', type=str, 
                        default='/home/yeskey525/Research_CODE/experiments/golden_answers/tree_json',
                        help='Directory to save the JSON tree files')
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load the existing ISO-3 tree with images
    iso3_tree = load_json_file(args.iso3_tree)
    if not iso3_tree:
        print(f"Error: Could not load ISO-3 tree from {args.iso3_tree}")
        return False
    
    # Process iso5_chunk.pkl if it exists
    iso5_output = os.path.join(args.output_dir, 'iso5_tree.json')
    if os.path.exists(args.iso5):
        print(f"Processing ISO-5 data from {args.iso5}")
        iso5_data = load_pickle_file(args.iso5)
        if iso5_data:
            # Handle different data structures
            iso5_tree = None
            
            try:
                # Try to convert the data to a dictionary if it has a tree-like structure
                if hasattr(iso5_data, 'children') or hasattr(iso5_data, 'text') or hasattr(iso5_data, 'id'):
                    print("ISO-5 data appears to be a tree-like object")
                    iso5_tree = convert_node_to_dict(iso5_data)
                # If it's a list, create a new root node and add each item as a child
                elif isinstance(iso5_data, list):
                    print("ISO-5 data is a list, creating a new root node")
                    iso5_tree = {
                        'text': 'ISO-5 Root',
                        'node_type': 'root',
                        'id': 'iso5_root',
                        'children': []
                    }
                    # Add each item in the list as a child
                    for i, item in enumerate(iso5_data):
                        # If the item is a dictionary or has attributes
                        if isinstance(item, dict):
                            # Use the item directly if it has the expected structure
                            iso5_tree['children'].append(item)
                        elif hasattr(item, '__dict__'):
                            # Try to convert the item to a dictionary
                            child = convert_node_to_dict(item)
                            if child:
                                iso5_tree['children'].append(child)
                        else:
                            # Create a simple node for the item
                            iso5_tree['children'].append({
                                'text': str(item),
                                'node_type': 'unknown',
                                'id': f'iso5_item_{i}'
                            })
                # If it's a dictionary, use it directly if it has the expected structure
                elif isinstance(iso5_data, dict):
                    print("ISO-5 data is a dictionary")
                    if 'text' in iso5_data or 'name' in iso5_data:
                        iso5_tree = iso5_data
                    else:
                        # Create a new root node and add the dictionary as its attributes
                        iso5_tree = {
                            'text': 'ISO-5 Root',
                            'node_type': 'root',
                            'id': 'iso5_root',
                            'metadata': iso5_data
                        }
                else:
                    print(f"ISO-5 data has an unexpected type: {type(iso5_data)}")
                    # Create a simple tree with the data as a string
                    iso5_tree = {
                        'text': 'ISO-5 Root',
                        'node_type': 'root',
                        'id': 'iso5_root',
                        'metadata': {
                            'raw_data': str(iso5_data)
                        }
                    }
            except Exception as e:
                print(f"Error converting ISO-5 data to tree: {e}")
                import traceback
                traceback.print_exc()
            
            if iso5_tree:
                save_json_tree(iso5_tree, iso5_output)
                print(f"Successfully saved ISO-5 tree to {iso5_output}")
            else:
                print("Failed to create ISO-5 tree")
                return False
        else:
            print(f"Error: Could not load ISO-5 data from {args.iso5}")
            return False
    else:
        print(f"Error: ISO-5 pickle file {args.iso5} not found")
        return False
    
    # Load the ISO-5 tree from the saved JSON file
    iso5_tree = load_json_file(iso5_output)
    if not iso5_tree:
        print(f"Error: Could not load ISO-5 tree from {iso5_output}")
        return False
    
    # Combine the ISO-3 and ISO-5 trees
    combined_output = os.path.join(args.output_dir, 'combined_tree.json')
    print("Combining ISO-3 and ISO-5 trees")
    
    # Create a combined tree with both ISO-3 and ISO-5 as children
    combined_tree = {
        'text': 'Root',
        'node_type': 'root',
        'id': 'combined_root',
        'children': [
            iso3_tree,
            iso5_tree
        ]
    }
    
    save_json_tree(combined_tree, combined_output)
    print(f"Combined tree saved to {combined_output}")
    
    print("Tree conversion and merging completed")
    return True

if __name__ == "__main__":
    main() 