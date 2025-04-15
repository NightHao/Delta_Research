import csv
import json
import pickle
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

def build_tree(content, source_name=None):
    """
    Build a tree structure from the content chunks
    This function is from create_chunks.ipynb
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

def merge_trees(trees):
    """
    Merge multiple trees into a single tree
    """
    if not trees:
        return None
    
    # Use the first tree as base
    merged_root = trees[0]
    
    # If there's only one tree, return it directly
    if len(trees) == 1:
        return merged_root
    
    # For each additional tree, merge its children into the base tree
    for tree_idx in range(1, len(trees)):
        other_tree = trees[tree_idx]
        
        # Merge all children from the other tree root into merged root
        for child in other_tree.children:
            # Clone the child and its descendants
            child.parent = merged_root
    
    return merged_root

def load_content_and_build_merged_tree(pickle_file_paths):
    """
    Load content from multiple pickle files and build a merged tree
    """
    trees = []
    
    for i, pickle_path in enumerate(pickle_file_paths):
        source_name = f"source_{i+1}"
        
        try:
            with open(pickle_path, 'rb') as f:
                content = pickle.load(f)
                print(f"Successfully loaded content from {pickle_path}")
                
                # Build tree for this content
                tree = build_tree(content, source_name)
                trees.append(tree)
                print(f"Built tree for {pickle_path} with {len(tree.descendants)} nodes")
                
        except Exception as e:
            print(f"Error loading content from {pickle_path}: {e}")
    
    if not trees:
        print("No trees were built. Exiting.")
        return None
    
    # Merge all trees
    merged_tree = merge_trees(trees)
    print(f"Created merged tree with {len(merged_tree.descendants)} total nodes")
    
    return merged_tree

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

def find_node_in_tree(text, tree, similarity_threshold=0.85):
    """
    Find a node in the tree that contains the given text
    Returns the node if found, None otherwise
    """
    if not text:
        return None
    
    text = clean_text(text)
    
    # Search through all nodes in the tree
    for node in tree.descendants:
        node_text = clean_text(node.name)
        
        # Check for exact match
        if text == node_text:
            return node
        
        # Check for high similarity
        if similar(text, node_text) > similarity_threshold:
            return node
        
        # Check for substring match (only for longer texts to avoid false matches)
        if len(text) > 10 and text in node_text:
            return node
    
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
    Get the text of the parent node
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
    
    for question_info in questions_data:
        question_result = {
            'question_number': question_info['question_number'],
            'question': question_info['question'],
            'entity': question_info['entity'],
            'gathered_information': []
        }
        
        # Process the table information
        if question_info['table']:
            table_texts = question_info['table'].split('\n')
            for table_text in table_texts:
                table_text = table_text.strip()
                if not table_text:
                    continue
                
                table_node = find_node_in_tree(table_text, tree)
                if table_node:
                    parent_text = get_parent_text(table_node)
                    siblings = get_all_siblings(table_node)
                    ancestors = get_ancestors(table_node)
                    source = getattr(table_node, 'source', 'Unknown')
                    
                    question_result['gathered_information'].append({
                        'source': 'table',
                        'text': table_text,
                        'parent_text': parent_text,
                        'siblings': siblings,
                        'ancestors': ancestors,
                        'section_id': getattr(table_node, 'section_id', 'None'),
                        'data_source': source
                    })
        
        # Process the figure information
        if question_info['figure']:
            for figure_text in question_info['figure'].split('\n'):
                figure_text = figure_text.strip()
                if not figure_text:
                    continue
                
                figure_node = find_node_in_tree(figure_text, tree)
                if figure_node:
                    parent_text = get_parent_text(figure_node)
                    siblings = get_all_siblings(figure_node)
                    ancestors = get_ancestors(figure_node)
                    source = getattr(figure_node, 'source', 'Unknown')
                    
                    question_result['gathered_information'].append({
                        'source': 'figure',
                        'text': figure_text,
                        'parent_text': parent_text,
                        'siblings': siblings,
                        'ancestors': ancestors,
                        'section_id': getattr(figure_node, 'section_id', 'None'),
                        'data_source': source
                    })
        
        # Process the answer information
        for answer in question_info['answers']:
            if not answer:
                continue
                
            answer_node = find_node_in_tree(answer, tree)
            if answer_node:
                parent_text = get_parent_text(answer_node)
                siblings = get_all_siblings(answer_node)
                ancestors = get_ancestors(answer_node)
                source = getattr(answer_node, 'source', 'Unknown')
                
                question_result['gathered_information'].append({
                    'source': 'answer',
                    'text': answer,
                    'parent_text': parent_text,
                    'siblings': siblings,
                    'ancestors': ancestors,
                    'section_id': getattr(answer_node, 'section_id', 'None'),
                    'data_source': source
                })
            else:
                # Include the answer even if it's not found in the tree
                question_result['gathered_information'].append({
                    'source': 'answer',
                    'text': answer,
                    'parent_text': 'CSV file',
                    'siblings': [],
                    'ancestors': ['CSV Data'],
                    'section_id': 'CSV',
                    'data_source': 'CSV'
                })
        
        results.append(question_result)
    
    return results

def organize_by_source(gathered_info):
    """
    Organize gathered information by source type for better readability
    """
    organized = {
        'table_info': [],
        'figure_info': [],
        'answer_info': []
    }
    
    for info in gathered_info:
        source = info['source']
        if source == 'table':
            organized['table_info'].append(info)
        elif source == 'figure':
            organized['figure_info'].append(info)
        elif source == 'answer':
            organized['answer_info'].append(info)
    
    return organized

def enrich_results(results):
    """
    Enrich the results with additional metadata and organization
    """
    enriched_results = []
    
    for question_result in results:
        enriched = {
            'question_number': question_result['question_number'],
            'question': question_result['question'],
            'entity': question_result['entity'],
            'organized_information': organize_by_source(question_result['gathered_information']),
            'all_information': question_result['gathered_information']
        }
        
        # Count the information pieces
        count_info = {
            'total_pieces': len(question_result['gathered_information']),
            'table_pieces': len(enriched['organized_information']['table_info']),
            'figure_pieces': len(enriched['organized_information']['figure_info']),
            'answer_pieces': len(enriched['organized_information']['answer_info'])
        }
        
        # Count by data source
        data_sources = {}
        for info in question_result['gathered_information']:
            source = info.get('data_source', 'Unknown')
            if source not in data_sources:
                data_sources[source] = 0
            data_sources[source] += 1
        
        enriched['count_info'] = count_info
        enriched['data_sources'] = data_sources
        enriched_results.append(enriched)
    
    return enriched_results

def create_summary(results):
    """
    Create a summary of information gathered across all questions
    """
    summary = {
        'total_questions': len(results),
        'total_information_pieces': sum(r['count_info']['total_pieces'] for r in results),
        'questions_with_no_information': [
            r['question_number'] for r in results if r['count_info']['total_pieces'] == 0
        ],
        'questions_by_information_count': {},
        'source_distribution': {
            'table': sum(r['count_info']['table_pieces'] for r in results),
            'figure': sum(r['count_info']['figure_pieces'] for r in results),
            'answer': sum(r['count_info']['answer_pieces'] for r in results)
        },
        'data_source_distribution': {}
    }
    
    # Count information by data source
    for result in results:
        for source, count in result.get('data_sources', {}).items():
            if source not in summary['data_source_distribution']:
                summary['data_source_distribution'][source] = 0
            summary['data_source_distribution'][source] += count
    
    # Group questions by information count
    for result in results:
        count = result['count_info']['total_pieces']
        if count not in summary['questions_by_information_count']:
            summary['questions_by_information_count'][count] = []
        summary['questions_by_information_count'][count].append(result['question_number'])
    
    return summary

def save_results(results, output_file):
    """
    Save the gathered information to a JSON file
    """
    # Enrich the results first
    enriched_results = enrich_results(results)
    
    # Create a summary
    summary = create_summary(enriched_results)
    
    # Add the summary to the output
    final_output = {
        'summary': summary,
        'questions': enriched_results
    }
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(final_output, f, indent=2, ensure_ascii=False)
        print(f"Results successfully saved to {output_file}")
    except Exception as e:
        print(f"Error saving results to {output_file}: {e}")

def main():
    # Check for debug flag
    debug_mode = "--debug" in sys.argv
    if debug_mode:
        print("Running in debug mode")
    
    # Paths - use absolute paths to ensure files are found
    # Adjust these paths according to your actual directory structure
    pickle_file_paths = [
        '/home/yeskey525/Research_CODE/experiments/golden_answers/ISO_15118-3_chunk.pkl',
        '/home/yeskey525/Research_CODE/experiments/golden_answers/iso5_chunk.pkl',
        '/home/yeskey525/Research_CODE/experiments/golden_answers/create_chunks.pkl'  # Replace with your actual file name
    ]
    
    # Check if the second file exists, and if not, inform the user
    for path in pickle_file_paths:
        if not os.path.exists(path):
            print(f"Warning: Pickle file {path} does not exist. It will be skipped.")
    
    # Filter out non-existent files
    pickle_file_paths = [path for path in pickle_file_paths if os.path.exists(path)]
    
    if not pickle_file_paths:
        print("Error: No valid pickle files found. Exiting.")
        return
    
    csv_file_path = '/home/yeskey525/Research_CODE/experiments/golden_answers/thesis_experiment_result_update.csv'
    output_file = '/home/yeskey525/Research_CODE/experiments/golden_answers/gathered_information.json'
    
    # Load content and build the merged tree
    tree = load_content_and_build_merged_tree(pickle_file_paths)
    if not tree:
        print("Failed to build merged tree. Exiting.")
        return
    
    print(f"Successfully built merged tree from {len(pickle_file_paths)} pickle files")
    
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