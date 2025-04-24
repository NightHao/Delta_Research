import json
import re
import os
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

def extract_tables_figures(answer_text):
    """
    Extract the specific tables and figures mentioned in a text.
    Returns sets of unique table and figure references.
    """
    # Pattern for tables: "Table X" or "Table X.Y" or "Tables X and Y"
    table_pattern = r'Table\s+[A-Za-z0-9.-]+|Tables\s+[A-Za-z0-9.-]+'
    # Pattern for figures: "Figure X" or "Figure X.Y" or "Figures X and Y"
    figure_pattern = r'Figure\s+[A-Za-z0-9.-]+|Figures\s+[A-Za-z0-9.-]+'
    
    # Find all references
    tables_raw = re.findall(table_pattern, answer_text)
    figures_raw = re.findall(figure_pattern, answer_text)
    
    # Process tables to handle plural forms
    tables = set()
    for table in tables_raw:
        if table.startswith("Tables"):
            # Try to extract multiple table references
            multi_tables = re.findall(r'[0-9]+\.[0-9]+|[0-9]+', table)
            if multi_tables:
                for t in multi_tables:
                    tables.add(f"Table {t}")
            else:
                # If we can't extract specific numbers, just add as is
                tables.add(table)
        else:
            tables.add(table)
    
    # Process figures to handle plural forms
    figures = set()
    for figure in figures_raw:
        if figure.startswith("Figures"):
            # Try to extract multiple figure references
            multi_figures = re.findall(r'[0-9]+\.[0-9]+|[0-9]+', figure)
            if multi_figures:
                for f in multi_figures:
                    figures.add(f"Figure {f}")
            else:
                # If we can't extract specific numbers, just add as is
                figures.add(figure)
        else:
            figures.add(figure)
    
    return tables, figures

def analyze_json_file(file_path):
    """
    Analyze a JSON file containing questions and answers.
    Returns a dictionary with questions as keys and sets of tables/figures as values.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    results = {}
    
    # Handle different JSON formats
    if 'qa_pairs' in data:
        # QA.json format
        for qa_pair in data['qa_pairs']:
            question = qa_pair['question']
            answer = qa_pair['answer']
            tables, figures = extract_tables_figures(answer)
            results[question] = {'tables': tables, 'figures': figures}
    else:
        # o1_answers.json format
        for question, answer in data.items():
            tables, figures = extract_tables_figures(answer)
            results[question] = {'tables': tables, 'figures': figures}
    
    return results

def compare_references(file_results):
    """
    Compare table and figure references across files for common questions.
    Returns a dictionary with comparison data.
    """
    # Find common questions across all files
    all_questions = set()
    for results in file_results.values():
        all_questions.update(results.keys())
    
    comparison = {}
    
    for question in all_questions:
        comparison[question] = {'tables': {}, 'figures': {}}
        
        # Compare tables and figures for each file
        for file_path, results in file_results.items():
            file_name = os.path.basename(file_path)
            
            if question in results:
                comparison[question]['tables'][file_name] = results[question]['tables']
                comparison[question]['figures'][file_name] = results[question]['figures']
            else:
                comparison[question]['tables'][file_name] = set()
                comparison[question]['figures'][file_name] = set()
    
    return comparison

def generate_statistics(comparison, file_paths):
    """
    Generate statistics from the comparison data.
    Returns statistics about unique and common references.
    """
    file_names = [os.path.basename(path) for path in file_paths]
    if len(file_names) < 2:
        print("Need at least two files for comparison")
        return None
    
    file1 = file_names[0]
    file2 = file_names[1]
    
    stats = {}
    
    # Overall statistics
    all_tables_file1 = set()
    all_tables_file2 = set()
    all_figures_file1 = set()
    all_figures_file2 = set()
    
    # Per-question statistics
    for question, data in comparison.items():
        tables_file1 = data['tables'].get(file1, set())
        tables_file2 = data['tables'].get(file2, set())
        figures_file1 = data['figures'].get(file1, set())
        figures_file2 = data['figures'].get(file2, set())
        
        # Update overall statistics
        all_tables_file1.update(tables_file1)
        all_tables_file2.update(tables_file2)
        all_figures_file1.update(figures_file1)
        all_figures_file2.update(figures_file2)
        
        # Calculate differences for this question
        tables_only_file1 = tables_file1 - tables_file2
        tables_only_file2 = tables_file2 - tables_file1
        tables_both = tables_file1.intersection(tables_file2)
        
        figures_only_file1 = figures_file1 - figures_file2
        figures_only_file2 = figures_file2 - figures_file1
        figures_both = figures_file1.intersection(figures_file2)
        
        # Store statistics for this question
        stats[question] = {
            'tables': {
                f'only_{file1}': tables_only_file1,
                f'only_{file2}': tables_only_file2,
                'both': tables_both,
                f'count_{file1}': len(tables_file1),
                f'count_{file2}': len(tables_file2),
                'count_both': len(tables_both)
            },
            'figures': {
                f'only_{file1}': figures_only_file1,
                f'only_{file2}': figures_only_file2,
                'both': figures_both,
                f'count_{file1}': len(figures_file1),
                f'count_{file2}': len(figures_file2),
                'count_both': len(figures_both)
            }
        }
    
    # Calculate overall statistics
    tables_only_file1_overall = all_tables_file1 - all_tables_file2
    tables_only_file2_overall = all_tables_file2 - all_tables_file1
    tables_both_overall = all_tables_file1.intersection(all_tables_file2)
    
    figures_only_file1_overall = all_figures_file1 - all_figures_file2
    figures_only_file2_overall = all_figures_file2 - all_figures_file1
    figures_both_overall = all_figures_file1.intersection(all_figures_file2)
    
    stats['overall'] = {
        'tables': {
            f'only_{file1}': tables_only_file1_overall,
            f'only_{file2}': tables_only_file2_overall,
            'both': tables_both_overall,
            f'count_{file1}': len(all_tables_file1),
            f'count_{file2}': len(all_tables_file2),
            'count_both': len(tables_both_overall)
        },
        'figures': {
            f'only_{file1}': figures_only_file1_overall,
            f'only_{file2}': figures_only_file2_overall,
            'both': figures_both_overall,
            f'count_{file1}': len(all_figures_file1),
            f'count_{file2}': len(all_figures_file2),
            'count_both': len(figures_both_overall)
        }
    }
    
    return stats

def visualize_differences(stats, file_paths, output_dir="./figures"):
    """
    Create visualizations of the differences between files.
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    file_names = [os.path.basename(path) for path in file_paths]
    if len(file_names) < 2:
        print("Need at least two files for comparison")
        return
    
    file1 = file_names[0]
    file2 = file_names[1]
    
    # Plot 1: Overall comparison of tables and figures
    plt.figure(figsize=(12, 6))
    
    # Tables data
    tables_data = [
        len(stats['overall']['tables'][f'only_{file1}']),
        len(stats['overall']['tables'][f'only_{file2}']),
        len(stats['overall']['tables']['both'])
    ]
    
    # Figures data
    figures_data = [
        len(stats['overall']['figures'][f'only_{file1}']),
        len(stats['overall']['figures'][f'only_{file2}']),
        len(stats['overall']['figures']['both'])
    ]
    
    x = np.arange(3)
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, tables_data, width, label='Tables')
    rects2 = ax.bar(x + width/2, figures_data, width, label='Figures')
    
    ax.set_ylabel('Count')
    ax.set_title('Overall Comparison of Table and Figure References')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Only in {file1}', f'Only in {file2}', 'In Both Files'])
    ax.legend()
    
    # Add counts on bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height}',
                        xy=(rect.get_x() + rect.get_width()/2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
    
    autolabel(rects1)
    autolabel(rects2)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/overall_comparison.png")
    plt.close()
    
    # Plot 2: Per-question comparison - Only include questions with differences
    # Collect data for questions with differences
    questions_with_diff = []
    tables_diff_counts = []
    figures_diff_counts = []
    
    for question, data in stats.items():
        if question == 'overall':
            continue
        
        # Check if there are differences in tables or figures
        tables_diff = (data['tables'][f'count_{file1}'] != data['tables'][f'count_{file2}'] or 
                      len(data['tables'][f'only_{file1}']) > 0 or 
                      len(data['tables'][f'only_{file2}']) > 0)
        
        figures_diff = (data['figures'][f'count_{file1}'] != data['figures'][f'count_{file2}'] or 
                       len(data['figures'][f'only_{file1}']) > 0 or 
                       len(data['figures'][f'only_{file2}']) > 0)
        
        if tables_diff or figures_diff:
            questions_with_diff.append(question)
            tables_diff_counts.append(
                abs(data['tables'][f'count_{file1}'] - data['tables'][f'count_{file2}'])
            )
            figures_diff_counts.append(
                abs(data['figures'][f'count_{file1}'] - data['figures'][f'count_{file2}'])
            )
    
    # Sort questions by the total difference
    # Display all questions with differences without sorting
    if questions_with_diff:
        # Shorten question text for better display
        short_questions = [q[:30] + '...' if len(q) > 30 else q for q in questions_with_diff]
        
        # Create horizontal bar chart
        fig, ax = plt.subplots(figsize=(12, 8))
        y_pos = np.arange(len(short_questions))
        
        ax.barh(y_pos - 0.2, tables_diff_counts, 0.4, label='Tables Difference')
        ax.barh(y_pos + 0.2, figures_diff_counts, 0.4, label='Figures Difference')
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(short_questions)
        ax.invert_yaxis()  # Labels read top-to-bottom
        ax.set_xlabel('Absolute Difference in Count')
        ax.set_title('Questions with Differences in Table and Figure References')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/questions_with_differences.png")
        plt.close()
    
    # Plot 3: Distribution of differences
    plt.figure(figsize=(10, 6))
    
    # Calculate difference in counts for all questions
    table_diffs = []
    figure_diffs = []
    
    for question, data in stats.items():
        if question != 'overall':
            table_diffs.append(data['tables'][f'count_{file1}'] - data['tables'][f'count_{file2}'])
            figure_diffs.append(data['figures'][f'count_{file1}'] - data['figures'][f'count_{file2}'])
    
    plt.hist(table_diffs, bins=range(min(table_diffs)-1, max(table_diffs)+2), alpha=0.5, label='Tables')
    plt.hist(figure_diffs, bins=range(min(figure_diffs)-1, max(figure_diffs)+2), alpha=0.5, label='Figures')
    
    plt.xlabel('Difference in Count (File1 - File2)')
    plt.ylabel('Number of Questions')
    plt.title('Distribution of Differences in Table and Figure Counts')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/difference_distribution.png")
    plt.close()

def print_detailed_report(stats, file_paths):
    """
    Print a detailed report of the comparison statistics.
    """
    file_names = [os.path.basename(path) for path in file_paths]
    if len(file_names) < 2:
        print("Need at least two files for comparison")
        return
    
    file1 = file_names[0]
    file2 = file_names[1]
    
    print("\n" + "="*80)
    print(f"DETAILED COMPARISON REPORT BETWEEN {file1} AND {file2}")
    print("="*80)
    
    # Print overall statistics
    print("\nOVERALL STATISTICS:")
    print("-"*40)
    print(f"Tables only in {file1}: {len(stats['overall']['tables'][f'only_{file1}'])}")
    print(f"Tables only in {file2}: {len(stats['overall']['tables'][f'only_{file2}'])}")
    print(f"Tables in both files: {len(stats['overall']['tables']['both'])}")
    print(f"Total unique tables in {file1}: {stats['overall']['tables'][f'count_{file1}']}")
    print(f"Total unique tables in {file2}: {stats['overall']['tables'][f'count_{file2}']}")
    
    print("\n")
    print(f"Figures only in {file1}: {len(stats['overall']['figures'][f'only_{file1}'])}")
    print(f"Figures only in {file2}: {len(stats['overall']['figures'][f'only_{file2}'])}")
    print(f"Figures in both files: {len(stats['overall']['figures']['both'])}")
    print(f"Total unique figures in {file1}: {stats['overall']['figures'][f'count_{file1}']}")
    print(f"Total unique figures in {file2}: {stats['overall']['figures'][f'count_{file2}']}")
    
    # Find questions with the biggest differences
    questions_by_table_diff = sorted(
        [(q, abs(stats[q]['tables'][f'count_{file1}'] - stats[q]['tables'][f'count_{file2}']))
         for q in stats if q != 'overall'],
        key=lambda x: x[1],
        reverse=True
    )
    
    questions_by_figure_diff = sorted(
        [(q, abs(stats[q]['figures'][f'count_{file1}'] - stats[q]['figures'][f'count_{file2}']))
         for q in stats if q != 'overall'],
        key=lambda x: x[1],
        reverse=True
    )
    
    # Print top questions with biggest differences
    print("\nTOP 5 QUESTIONS WITH BIGGEST TABLE DIFFERENCES:")
    print("-"*40)
    for i, (question, diff) in enumerate(questions_by_table_diff[:5], 1):
        print(f"{i}. '{question[:50]}...' (Difference: {diff})")
        print(f"   Tables in {file1}: {stats[question]['tables'][f'count_{file1}']}")
        print(f"   Tables in {file2}: {stats[question]['tables'][f'count_{file2}']}")
        
        # List the specific tables that differ
        if stats[question]['tables'][f'only_{file1}']:
            print(f"   Tables only in {file1}:")
            for table in stats[question]['tables'][f'only_{file1}']:
                print(f"     - {table}")
        
        if stats[question]['tables'][f'only_{file2}']:
            print(f"   Tables only in {file2}:")
            for table in stats[question]['tables'][f'only_{file2}']:
                print(f"     - {table}")
        print()
    
    print("\nTOP 5 QUESTIONS WITH BIGGEST FIGURE DIFFERENCES:")
    print("-"*40)
    for i, (question, diff) in enumerate(questions_by_figure_diff[:5], 1):
        print(f"{i}. '{question[:50]}...' (Difference: {diff})")
        print(f"   Figures in {file1}: {stats[question]['figures'][f'count_{file1}']}")
        print(f"   Figures in {file2}: {stats[question]['figures'][f'count_{file2}']}")
        
        # List the specific figures that differ
        if stats[question]['figures'][f'only_{file1}']:
            print(f"   Figures only in {file1}:")
            for figure in stats[question]['figures'][f'only_{file1}']:
                print(f"     - {figure}")
        
        if stats[question]['figures'][f'only_{file2}']:
            print(f"   Figures only in {file2}:")
            for figure in stats[question]['figures'][f'only_{file2}']:
                print(f"     - {figure}")
        print()

def main():
    # List of JSON files to analyze
    json_files = ['./complete_golden_answers.json', './o4mini_golden_ans.json']
    
    # Store results per file
    file_results = {}
    
    # Analyze each file
    for file_path in json_files:
        if os.path.exists(file_path):
            print(f"\nAnalyzing {file_path}...")
            results = analyze_json_file(file_path)
            file_results[file_path] = results
            
            # Print summary for this file
            total_tables = sum(len(result['tables']) for result in results.values())
            total_figures = sum(len(result['figures']) for result in results.values())
            print(f"Total in {file_path}: {total_tables} tables, {total_figures} figures")
        else:
            print(f"File {file_path} not found.")
    
    # Compare references across files
    comparison = compare_references(file_results)
    
    # Generate statistics
    stats = generate_statistics(comparison, json_files)
    
    # Print detailed report
    print_detailed_report(stats, json_files)
    
    # Create visualizations
    visualize_differences(stats, json_files)
    
    print("\nAnalysis complete! Visualizations saved to './figures' directory.")

if __name__ == "__main__":
    main() 