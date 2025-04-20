#!/usr/bin/env python3
"""
Script to gather information using the pre-built iso3_tree_with_images.json tree
rather than building the tree from pickle files.
"""

import csv
import json
import re
import sys
import os
from collections import defaultdict
from difflib import SequenceMatcher
from anytree import Node, RenderTree

def print_csv_headers(csv_file_path):
    """Print the headers and first few rows of a CSV file for debugging"""
    try:
        with open(csv_file_path, 'r', encoding='utf-8') as file:
            reader = csv.reader(file)
            headers = next(reader)
            print("\nCSV Headers:", headers)
            
            print("\nFirst 3 rows:")
            for i, row in enumerate(reader):
                print(f"Row {i+1}:", row)
                if i >= 2:  # Just show the first 3 rows
                    break
    except Exception as e:
        print(f"Error reading CSV file: {e}")

def load_csv_data(csv_file_path, debug=False):
    """
    Load the CSV data containing questions and their answers
    """
    if debug:
        print_csv_headers(csv_file_path)
    
    questions_data = []
    
    try:
        with open(csv_file_path, 'r', encoding='utf-8') as file:
            # Try different encodings if UTF-8 fails
            try:
                reader = csv.DictReader(file)
                headers = reader.fieldnames
                if not headers:
                    raise ValueError("No headers found in CSV file")
                
                print(f"CSV headers: {headers}")
            except UnicodeDecodeError:
                file.close()
                # Try with different encodings
                for encoding in ['latin-1', 'iso-8859-1', 'cp1252']:
                    try:
                        with open(csv_file_path, 'r', encoding=encoding) as f:
                            reader = csv.DictReader(f)
                            headers = reader.fieldnames
                            if headers:
                                print(f"Successfully read CSV with encoding: {encoding}")
                                break
                    except:
                        continue
                else:
                    raise ValueError("Could not decode CSV file with any attempted encoding")
            
            # Find the right column names - check for variations
            question_num_col = None
            question_col = None
            entity_col = None
            table_col = None
            figure_col = None
            
            # Look for question number column
            for possible_name in ['Question number', 'question number', 'Question Number', 'QuestionNumber', 'Question_number', 'question_number', 'Question', '#']:
                if possible_name in headers:
                    question_num_col = possible_name
                    break
            
            # Look for question text column
            for possible_name in ['Question', 'question', 'QuestionText', 'Question Text', 'question_text', 'Question_text']:
                if possible_name in headers:
                    question_col = possible_name
                    break
            
            # Look for entity column
            for possible_name in ['Entity', 'entity', 'Subject', 'subject']:
                if possible_name in headers:
                    entity_col = possible_name
                    break
            
            # Look for table column
            for possible_name in ['Table', 'table', 'Tables', 'tables']:
                if possible_name in headers:
                    table_col = possible_name
                    break
            
            # Look for figure column
            for possible_name in ['Figure', 'figure', 'Figures', 'figures']:
                if possible_name in headers:
                    figure_col = possible_name
                    break
            
            # If we couldn't find the right columns, use the first ones
            if question_num_col is None and len(headers) > 0:
                question_num_col = headers[0]
                print(f"Warning: Could not find question number column. Using first column: {question_num_col}")
            
            if question_col is None and len(headers) > 1:
                question_col = headers[1]
                print(f"Warning: Could not find question text column. Using second column: {question_col}")
            
            print(f"Using columns: Question Number='{question_num_col}', Question='{question_col}', Entity='{entity_col}', Table='{table_col}', Figure='{figure_col}'")
            
            # Reset file position and reader to process rows
            file.seek(0)
            next(file)  # Skip header row
            reader = csv.DictReader(file, fieldnames=headers)
            
            row_count = 0
            for row in reader:
                row_count += 1
                # Get question number and text
                question_number = row.get(question_num_col, "")
                question_text = row.get(question_col, "")
                entity = row.get(entity_col, "") if entity_col else ""
                table = row.get(table_col, "") if table_col else ""
                figure = row.get(figure_col, "") if figure_col else ""
                
                # Skip empty rows
                if not question_number and not question_text:
                    continue
                    
                question_info = {
                    'question_number': question_number,
                    'question': question_text,
                    'entity': entity,
                    'table': table,
                    'figure': figure,
                    'answers': []
                }
                
                # Collect all non-empty answers - try multiple naming patterns
                for i in range(1, 31):  # Answer1 to Answer30
                    # Try different possible naming formats for answer columns
                    answer_value = None
                    for key_pattern in [f'Answer{i}', f'answer{i}', f'Answer_{i}', f'answer_{i}', f'A{i}', f'a{i}']:
                        if key_pattern in row and row[key_pattern] and row[key_pattern].strip():
                            answer_value = row[key_pattern].strip()
                            break
                    
                    # If none of the patterns match, try generic detection
                    if answer_value is None:
                        # Look for any column that might be an answer column
                        for key in row:
                            if (('answer' in key.lower() or 'a' == key.lower()) and 
                                str(i) in key and row[key] and row[key].strip()):
                                answer_value = row[key].strip()
                                break
                    
                    if answer_value:
                        question_info['answers'].append(answer_value)
                
                questions_data.append(question_info)
                
                if debug and len(questions_data) <= 3:
                    print(f"Processed Question {question_number}: {question_text[:30]}... with {len(question_info['answers'])} answers")
            
            print(f"Loaded {len(questions_data)} questions from {row_count} rows in CSV")
    
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        if debug:
            import traceback
            traceback.print_exc()
    
    return questions_data

def json_to_anytree(json_tree, parent=None):
    """
    Convert a JSON tree structure to an anytree Node structure
    """
    # Prepare attributes dictionary
    attrs = {}
    for attr in ['path', 'depth', 'section_id', 'source', 'image_description']:
        if attr in json_tree:
            attrs[attr] = json_tree[attr]
    
    if parent is None:
        # Create the root node with attributes
        root = Node(json_tree['name'], **attrs)
        
        # Process children
        if 'children' in json_tree and json_tree['children']:
            for child in json_tree['children']:
                json_to_anytree(child, root)
        
        return root
    else:
        # Create a child node with attributes
        node = Node(json_tree['name'], parent=parent, **attrs)
        
        # Process children
        if 'children' in json_tree and json_tree['children']:
            for child in json_tree['children']:
                json_to_anytree(child, node)
        
        return node

def load_tree_from_json(json_file_path):
    """
    Load the tree from a JSON file
    """
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            tree_json = json.load(f)
            print(f"Successfully loaded tree from {json_file_path}")
            
            # Convert JSON tree to anytree
            tree = json_to_anytree(tree_json)
            print(f"Converted tree with {len(tree.descendants)} nodes")
            
            return tree
    except Exception as e:
        print(f"Error loading tree from {json_file_path}: {e}")
        return None

def similar(a, b):
    """
    Calculate string similarity ratio between two strings
    """
    return SequenceMatcher(None, a, b).ratio()

def clean_text(text):
    """
    Clean text for better matching by removing extra whitespace and normalizing
    """
    if not text:
        return ""
    # Replace multiple whitespace with a single space
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def find_node_in_tree(text, tree, similarity_threshold=0.85, debug=False):
    """
    Find a node in the tree that contains the given text
    Returns the node if found, None otherwise
    """
    if not text:
        return None
    
    text = clean_text(text)
    if debug:
        print(f"      Searching for text: '{text[:30]}...'")
    
    # Search through all nodes in the tree
    nodes_checked = 0
    for node in tree.descendants:
        nodes_checked += 1
        if nodes_checked % 100 == 0 and debug:
            print(f"      Checked {nodes_checked} nodes so far...")
        
        node_text = clean_text(node.name)
        
        # Check for exact match
        if text == node_text:
            if debug:
                print(f"      Found exact match after checking {nodes_checked} nodes")
            return node
        
        # Check for high similarity
        similarity = similar(text, node_text)
        if similarity > similarity_threshold:
            if debug:
                print(f"      Found similar match ({similarity:.2f}) after checking {nodes_checked} nodes")
            return node
        
        # Check for substring match (only for longer texts to avoid false matches)
        if len(text) > 10 and text in node_text:
            if debug:
                print(f"      Found substring match after checking {nodes_checked} nodes")
            return node
    
    if debug:
        print(f"      No match found after checking {nodes_checked} nodes")
    return None

def get_node_text(node):
    """
    Extract text from a node
    """
    if node:
        return node.name
    return "No text found"

def get_all_siblings(node):
    """
    Get all siblings of a node (other children of the same parent)
    """
    if not node or not node.parent:
        return []
    
    # Get all children of the parent except the node itself
    return [child.name for child in node.parent.children if child != node]

def get_parent_text(node):
    """
    Extract text from a node's parent
    """
    if node and node.parent:
        return node.parent.name
    return "No parent found"

def get_ancestors(node, max_depth=3):
    """
    Get the ancestors of a node up to a certain depth
    """
    ancestors = []
    current = node
    depth = 0
    
    while current.parent and depth < max_depth:
        current = current.parent
        ancestors.append(current.name)
        depth += 1
    
    return ancestors

def gather_information(questions_data, tree):
    """
    Gather comprehensive information for each question by finding nodes in the tree
    and collecting their parent nodes and siblings
    """
    results = []
    debug = "--debug" in sys.argv
    
    for i, question_info in enumerate(questions_data):
        print(f"Processing question {i+1}/{len(questions_data)}: {question_info['question_number']}")
        
        question_result = {
            'question_number': question_info['question_number'],
            'question': question_info['question'],
            'entity': question_info['entity'],
            'gathered_information': []
        }
        
        # Process the table information
        if question_info['table']:
            print(f"  Processing table information for question {i+1}")
            table_texts = question_info['table'].split('\n')
            for table_text in table_texts:
                table_text = table_text.strip()
                if not table_text:
                    continue
                
                print(f"    Looking for table node: {table_text[:50]}...")
                table_node = find_node_in_tree(table_text, tree, debug=debug)
                if table_node:
                    parent_text = get_parent_text(table_node)
                    siblings = get_all_siblings(table_node)
                    ancestors = get_ancestors(table_node)
                    source = getattr(table_node, 'source', 'Unknown')
                    
                    # Check if the node has image_description
                    image_description = None
                    if hasattr(table_node, 'image_description'):
                        image_description = getattr(table_node, 'image_description')
                    
                    info = {
                        'source': 'table',
                        'text': table_text,
                        'parent_text': parent_text,
                        'siblings': siblings,
                        'ancestors': ancestors,
                        'section_id': getattr(table_node, 'section_id', 'None'),
                        'data_source': source
                    }
                    
                    # Add image description if it exists
                    if image_description:
                        info['image_description'] = image_description
                    
                    question_result['gathered_information'].append(info)
                    print(f"    Found table node: {table_text[:50]}...")
                else:
                    print(f"    No match found for table: {table_text[:50]}...")
        
        # Process the figure information
        if question_info['figure']:
            print(f"  Processing figure information for question {i+1}")
            for figure_text in question_info['figure'].split('\n'):
                figure_text = figure_text.strip()
                if not figure_text:
                    continue
                
                print(f"    Looking for figure node: {figure_text[:50]}...")
                figure_node = find_node_in_tree(figure_text, tree, debug=debug)
                if figure_node:
                    parent_text = get_parent_text(figure_node)
                    siblings = get_all_siblings(figure_node)
                    ancestors = get_ancestors(figure_node)
                    source = getattr(figure_node, 'source', 'Unknown')
                    
                    # Check if the node has image_description
                    image_description = None
                    if hasattr(figure_node, 'image_description'):
                        image_description = getattr(figure_node, 'image_description')
                    
                    info = {
                        'source': 'figure',
                        'text': figure_text,
                        'parent_text': parent_text,
                        'siblings': siblings,
                        'ancestors': ancestors,
                        'section_id': getattr(figure_node, 'section_id', 'None'),
                        'data_source': source
                    }
                    
                    # Add image description if it exists
                    if image_description:
                        info['image_description'] = image_description
                    
                    question_result['gathered_information'].append(info)
                    print(f"    Found figure node: {figure_text[:50]}...")
                else:
                    print(f"    No match found for figure: {figure_text[:50]}...")
        
        # Process the answer information
        if question_info['answers']:
            print(f"  Processing {len(question_info['answers'])} answers for question {i+1}")
            for j, answer in enumerate(question_info['answers']):
                if not answer:
                    continue
                
                print(f"    Looking for answer node {j+1}/{len(question_info['answers'])}: {answer[:50]}...")
                answer_node = find_node_in_tree(answer, tree, debug=debug)
                if answer_node:
                    parent_text = get_parent_text(answer_node)
                    siblings = get_all_siblings(answer_node)
                    ancestors = get_ancestors(answer_node)
                    source = getattr(answer_node, 'source', 'Unknown')
                    
                    # Check if the node has image_description
                    image_description = None
                    if hasattr(answer_node, 'image_description'):
                        image_description = getattr(answer_node, 'image_description')
                    
                    info = {
                        'source': 'answer',
                        'text': answer,
                        'parent_text': parent_text,
                        'siblings': siblings,
                        'ancestors': ancestors,
                        'section_id': getattr(answer_node, 'section_id', 'None'),
                        'data_source': source
                    }
                    
                    # Add image description if it exists
                    if image_description:
                        info['image_description'] = image_description
                    
                    question_result['gathered_information'].append(info)
                    print(f"    Found answer node {j+1}: {answer[:50]}...")
                else:
                    print(f"    No match found for answer {j+1}: {answer[:50]}...")
        
        results.append(question_result)
        print(f"Completed question {i+1}/{len(questions_data)}")
        
        # Save intermittent results after each question to avoid losing progress
        if (i + 1) % 5 == 0 or i == len(questions_data) - 1:
            temp_output_file = f'/home/yeskey525/Research_CODE/experiments/golden_answers/gathered_information_with_images_partial_{i+1}.json'
            try:
                with open(temp_output_file, 'w', encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
                print(f"Saved partial results to {temp_output_file}")
            except Exception as e:
                print(f"Error saving partial results: {e}")
    
    return results

def save_results(results, output_file):
    """
    Save the gathered information to a JSON file
    """
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"Results saved to {output_file}")
    except Exception as e:
        print(f"Error saving results to {output_file}: {e}")

def main():
    # Check for debug flag
    debug_mode = "--debug" in sys.argv
    if debug_mode:
        print("Running in debug mode")
    
    # Check for --input parameter
    input_param = None
    for i, arg in enumerate(sys.argv):
        if arg == "--input" and i + 1 < len(sys.argv):
            input_param = sys.argv[i + 1]
            break
    
    # Paths - use absolute paths to ensure files are found
    json_tree_path = input_param if input_param else '/home/yeskey525/Research_CODE/experiments/golden_answers/iso3_tree_with_images.json'
    csv_file_path = '/home/yeskey525/Research_CODE/experiments/golden_answers/thesis_experiment_result_update.csv'
    output_file = '/home/yeskey525/Research_CODE/experiments/golden_answers/gathered_information_with_images.json'
    
    # Print the paths being used
    print(f"Using JSON tree: {json_tree_path}")
    print(f"Using CSV file: {csv_file_path}")
    print(f"Output will be saved to: {output_file}")
    
    # Check if files exist
    if not os.path.exists(json_tree_path):
        print(f"Error: JSON tree file {json_tree_path} does not exist. Exiting.")
        return
    
    if not os.path.exists(csv_file_path):
        print(f"Error: CSV file {csv_file_path} does not exist. Exiting.")
        return
    
    # Load the tree from JSON
    tree = load_tree_from_json(json_tree_path)
    if not tree:
        print("Failed to load tree from JSON. Exiting.")
        return
    
    print(f"Successfully loaded tree from {json_tree_path}")
    
    # Load the CSV data
    questions_data = load_csv_data(csv_file_path, debug=debug_mode)
    
    if not questions_data:
        print("No questions loaded from CSV. Exiting.")
        return
    
    print(f"Loaded {len(questions_data)} questions from CSV")
    
    # Gather information for each question
    results = gather_information(questions_data, tree)
    print(f"Gathered information for {len(results)} questions")
    
    # Save the results
    save_results(results, output_file)
    print(f"Information gathering complete. Results saved to {output_file}")
    
    # Provide a brief summary to console
    total_info = sum(len(r['gathered_information']) for r in results)
    print(f"Total information pieces gathered: {total_info}")
    print(f"Average information pieces per question: {total_info / len(results):.2f}")

if __name__ == "__main__":
    main() 