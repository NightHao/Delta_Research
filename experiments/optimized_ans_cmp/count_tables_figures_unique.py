#!/usr/bin/env python3
import json
import re
import os
import matplotlib.pyplot as plt
from collections import defaultdict

def analyze_gathered_information(file_path):
    """
    Analyze gathered_information.json file to identify and count tables and figures.
    """
    print(f"Analyzing {file_path}...")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        tables = []
        figures = []

        # Process all questions in the gathered information
        for question in data.get('questions', []):
            # Extract table information
            for table_info in question.get('organized_information', {}).get('table_info', []):
                if 'text' in table_info:
                    # Extract the table number from the text
                    match = re.search(r'Table\s+([A-Z0-9](?:\.\d+)*)', table_info['text'], re.IGNORECASE)
                    if match:
                        table_id = match.group(1)
                        tables.append({
                            'id': table_id,
                            'name': table_info['text'],
                            'source': table_info.get('source', '')
                        })
            
            # Extract figure information
            for figure_info in question.get('organized_information', {}).get('figure_info', []):
                if 'text' in figure_info:
                    # Extract the figure number from the text
                    match = re.search(r'Figure\s+([A-Z0-9](?:\.\d+)*)', figure_info['text'], re.IGNORECASE)
                    if match:
                        figure_id = match.group(1)
                        figures.append({
                            'id': figure_id,
                            'name': figure_info['text'],
                            'source': figure_info.get('source', '')
                        })
            
            # Also search in all_information array
            for item in question.get('all_information', []):
                if item['source'] == 'table' and 'text' in item:
                    match = re.search(r'Table\s+([A-Z0-9](?:\.\d+)*)', item['text'], re.IGNORECASE)
                    if match:
                        table_id = match.group(1)
                        tables.append({
                            'id': table_id,
                            'name': item['text'],
                            'source': item.get('source', '')
                        })
                elif item['source'] == 'figure' and 'text' in item:
                    match = re.search(r'Figure\s+([A-Z0-9](?:\.\d+)*)', item['text'], re.IGNORECASE)
                    if match:
                        figure_id = match.group(1)
                        figures.append({
                            'id': figure_id,
                            'name': item['text'],
                            'source': item.get('source', '')
                        })
        
        # Create unique lists based on IDs
        unique_tables = {}
        unique_figures = {}
        
        for table in tables:
            unique_tables[table['id']] = table
        
        for figure in figures:
            unique_figures[figure['id']] = figure
        
        return {
            'tables': list(unique_tables.values()),
            'figures': list(unique_figures.values())
        }
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return {'tables': [], 'figures': []}

def analyze_entity_graph(file_path):
    """
    Analyze entity_graph.json to identify and count tables and figures in the keys.
    """
    print(f"Analyzing {file_path}...")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    tables = []
    figures = []
    
    # Check all keys in the top-level dictionary
    for key in data.keys():
        table_match = re.search(r'TABLE\s+(\d+(?:\.\d+)*)', key, re.IGNORECASE)
        figure_match = re.search(r'FIGURE\s+(\d+(?:\.\d+)*)', key, re.IGNORECASE)
        
        if table_match:
            table_id = table_match.group(1)
            tables.append({
                'id': table_id,
                'name': key
            })
        
        if figure_match:
            figure_id = figure_match.group(1)
            figures.append({
                'id': figure_id,
                'name': key
            })
    
    # Create unique lists based on IDs
    unique_tables = {}
    unique_figures = {}
    
    for table in tables:
        unique_tables[table['id']] = table
    
    for figure in figures:
        unique_figures[figure['id']] = figure
    
    return {
        'tables': list(unique_tables.values()),
        'figures': list(unique_figures.values())
    }

def compare_results(gathered_results, entity_results):
    """Compare analysis results between the two files"""
    # Extract IDs for comparison
    gathered_table_ids = {item['id'] for item in gathered_results['tables']}
    gathered_figure_ids = {item['id'] for item in gathered_results['figures']}
    
    entity_table_ids = {item['id'] for item in entity_results['tables']}
    entity_figure_ids = {item['id'] for item in entity_results['figures']}
    
    # Find common and unique items
    common_tables = gathered_table_ids.intersection(entity_table_ids)
    common_figures = gathered_figure_ids.intersection(entity_figure_ids)
    
    gathered_only_tables = gathered_table_ids - entity_table_ids
    gathered_only_figures = gathered_figure_ids - entity_figure_ids
    
    entity_only_tables = entity_table_ids - gathered_table_ids
    entity_only_figures = entity_figure_ids - gathered_figure_ids
    
    return {
        'common_tables': common_tables,
        'common_figures': common_figures,
        'gathered_only_tables': gathered_only_tables,
        'gathered_only_figures': gathered_only_figures,
        'entity_only_tables': entity_only_tables,
        'entity_only_figures': entity_only_figures
    }

def save_results(results, filename):
    """Save the analysis results to a JSON file"""
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {filename}")

def print_results(results, title):
    """Print analysis results in a readable format"""
    print(f"\n{title}")
    print("=" * len(title))
    
    print(f"\nTables ({len(results['tables'])}):")
    for i, table in enumerate(sorted(results['tables'], key=lambda x: x['id']), 1):
        print(f"{i}. {table['id']}: {table['name'][:50]}...")
    
    print(f"\nFigures ({len(results['figures'])}):")
    for i, figure in enumerate(sorted(results['figures'], key=lambda x: x['id']), 1):
        print(f"{i}. {figure['id']}: {figure['name'][:50]}...")

def print_comparison(comparison):
    """Print the comparison results"""
    print("\nComparison Results")
    print("=================")
    
    print(f"\nCommon Tables ({len(comparison['common_tables'])}):")
    for i, table_id in enumerate(sorted(comparison['common_tables']), 1):
        print(f"{i}. Table {table_id}")
    
    print(f"\nCommon Figures ({len(comparison['common_figures'])}):")
    for i, figure_id in enumerate(sorted(comparison['common_figures']), 1):
        print(f"{i}. Figure {figure_id}")
    
    print(f"\nTables unique to Gathered Information ({len(comparison['gathered_only_tables'])}):")
    for i, table_id in enumerate(sorted(comparison['gathered_only_tables']), 1):
        print(f"{i}. Table {table_id}")
    
    print(f"\nFigures unique to Gathered Information ({len(comparison['gathered_only_figures'])}):")
    for i, figure_id in enumerate(sorted(comparison['gathered_only_figures']), 1):
        print(f"{i}. Figure {figure_id}")
    
    print(f"\nTables unique to Entity Graph ({len(comparison['entity_only_tables'])}):")
    for i, table_id in enumerate(sorted(comparison['entity_only_tables']), 1):
        print(f"{i}. Table {table_id}")
    
    print(f"\nFigures unique to Entity Graph ({len(comparison['entity_only_figures'])}):")
    for i, figure_id in enumerate(sorted(comparison['entity_only_figures']), 1):
        print(f"{i}. Figure {figure_id}")

def save_comparison_results(comparison, filename):
    """Save the comparison results to a JSON file"""
    # Convert sets to lists for JSON serialization
    serializable_comparison = {
        'common_tables': sorted(list(comparison['common_tables'])),
        'common_figures': sorted(list(comparison['common_figures'])),
        'gathered_only_tables': sorted(list(comparison['gathered_only_tables'])),
        'gathered_only_figures': sorted(list(comparison['gathered_only_figures'])),
        'entity_only_tables': sorted(list(comparison['entity_only_tables'])),
        'entity_only_figures': sorted(list(comparison['entity_only_figures']))
    }
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(serializable_comparison, f, indent=2)
    print(f"Comparison results saved to {filename}")

def visualize_comparison(comparison, gathered_results, entity_results, output_dir):
    """Create visualizations to show the comparison results"""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create bar chart for table counts
    plt.figure(figsize=(12, 6))
    
    # Set up the data
    labels = ['Gathered Information', 'Entity Graph', 'Common']
    tables_data = [
        len(gathered_results['tables']), 
        len(entity_results['tables']), 
        len(comparison['common_tables'])
    ]
    figures_data = [
        len(gathered_results['figures']), 
        len(entity_results['figures']), 
        len(comparison['common_figures'])
    ]
    
    # Set position of bars on x-axis
    bar_width = 0.35
    r1 = range(len(labels))
    r2 = [x + bar_width for x in r1]
    
    # Create bars
    plt.bar(r1, tables_data, width=bar_width, label='Tables', color='skyblue')
    plt.bar(r2, figures_data, width=bar_width, label='Figures', color='lightgreen')
    
    # Add labels and title
    plt.xlabel('Source')
    plt.ylabel('Count')
    plt.title('Tables and Figures Count Comparison')
    plt.xticks([r + bar_width/2 for r in range(len(labels))], labels)
    plt.legend()
    
    # Save the figure
    plt.savefig(os.path.join(output_dir, 'comparison_bar_chart.png'), bbox_inches='tight')
    plt.close()
    
    # Create pie charts
    # Pie chart for tables
    plt.figure(figsize=(10, 7))
    plt.subplot(1, 2, 1)
    table_sizes = [
        len(comparison['gathered_only_tables']),
        len(comparison['entity_only_tables']),
        len(comparison['common_tables'])
    ]
    labels = [
        f'Gathered Only\n({table_sizes[0]})', 
        f'Entity Only\n({table_sizes[1]})', 
        f'Common\n({table_sizes[2]})'
    ]
    colors = ['skyblue', 'lightgreen', 'lightcoral']
    plt.pie(table_sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    plt.axis('equal')
    plt.title('Tables Distribution')
    
    # Pie chart for figures
    plt.subplot(1, 2, 2)
    figure_sizes = [
        len(comparison['gathered_only_figures']),
        len(comparison['entity_only_figures']),
        len(comparison['common_figures'])
    ]
    labels = [
        f'Gathered Only\n({figure_sizes[0]})', 
        f'Entity Only\n({figure_sizes[1]})', 
        f'Common\n({figure_sizes[2]})'
    ]
    plt.pie(figure_sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    plt.axis('equal')
    plt.title('Figures Distribution')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pie_charts.png'), bbox_inches='tight')
    plt.close()
    
    print(f"Visualizations saved to {output_dir}")

def main():
    base_dir = "/home/yeskey525/Research_CODE/experiments/optimized_ans_cmp"
    gathered_file = os.path.join(base_dir, "gathered_information.json")
    entity_file = os.path.join(base_dir, "entity_graph.json")
    
    gathered_analysis_file = os.path.join(base_dir, "gathered_information_analysis.json")
    entity_analysis_file = os.path.join(base_dir, "entity_graph_analysis.json")
    comparison_file = os.path.join(base_dir, "tables_figures_comparison.json")
    visualization_dir = os.path.join(base_dir, "figures")
    
    # Run the analysis
    gathered_results = analyze_gathered_information(gathered_file)
    entity_results = analyze_entity_graph(entity_file)
    
    # Save analysis results
    save_results(gathered_results, gathered_analysis_file)
    save_results(entity_results, entity_analysis_file)
    
    # Print individual results
    print_results(gathered_results, "Gathered Information Analysis")
    print_results(entity_results, "Entity Graph Analysis")
    
    # Compare results
    comparison = compare_results(gathered_results, entity_results)
    print_comparison(comparison)
    
    # Save comparison results
    save_comparison_results(comparison, comparison_file)
    
    # Create visualizations
    visualize_comparison(comparison, gathered_results, entity_results, visualization_dir)
    
    # Print summary
    print("\nSummary:")
    print(f"Gathered Information: {len(gathered_results['tables'])} unique tables, {len(gathered_results['figures'])} unique figures")
    print(f"Entity Graph: {len(entity_results['tables'])} unique tables, {len(entity_results['figures'])} unique figures")
    print(f"Common: {len(comparison['common_tables'])} tables, {len(comparison['common_figures'])} figures")
    print(f"Unique to Gathered Information: {len(comparison['gathered_only_tables'])} tables, {len(comparison['gathered_only_figures'])} figures")
    print(f"Unique to Entity Graph: {len(comparison['entity_only_tables'])} tables, {len(comparison['entity_only_figures'])} figures")

if __name__ == "__main__":
    main() 