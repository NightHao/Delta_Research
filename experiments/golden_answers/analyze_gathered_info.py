import json
import os
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter, defaultdict

def load_gathered_info(json_file_path):
    """
    Load the gathered information from the JSON file
    """
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def generate_basic_stats(data):
    """
    Generate basic statistics from the gathered information
    """
    summary = data.get('summary', {})
    total_questions = summary.get('total_questions', 0)
    total_info_pieces = summary.get('total_information_pieces', 0)
    
    stats = {
        'total_questions': total_questions,
        'total_information_pieces': total_info_pieces,
        'avg_info_per_question': total_info_pieces / total_questions if total_questions > 0 else 0,
        'questions_with_no_info': len(summary.get('questions_with_no_information', [])),
        'source_distribution': summary.get('source_distribution', {})
    }
    
    return stats

def analyze_parent_child_relationships(data):
    """
    Analyze parent-child relationships in the gathered information
    """
    questions = data.get('questions', [])
    
    # Count occurrences of parent-child relationships
    parent_child_counts = Counter()
    
    # Track which questions have which relationships
    relationship_by_question = defaultdict(list)
    
    for question in questions:
        question_num = question.get('question_number', 'unknown')
        
        for info in question.get('all_information', []):
            parent_text = info.get('parent_text', 'None')
            node_text = info.get('text', 'Unknown')
            
            if parent_text != 'No parent found':
                # Use shortened text for readability
                parent_short = parent_text[:30] + "..." if len(parent_text) > 30 else parent_text
                node_short = node_text[:30] + "..." if len(node_text) > 30 else node_text
                
                relationship = f"{parent_short} -> {node_short}"
                parent_child_counts[relationship] += 1
                relationship_by_question[question_num].append(relationship)
    
    return {
        'parent_child_counts': dict(parent_child_counts.most_common(20)),
        'relationship_by_question': dict(relationship_by_question)
    }

def analyze_sibling_relationships(data):
    """
    Analyze sibling relationships in the gathered information
    """
    questions = data.get('questions', [])
    
    # Count number of siblings per information piece
    sibling_counts = []
    
    # Track which questions have which number of siblings
    siblings_by_question = defaultdict(list)
    
    for question in questions:
        question_num = question.get('question_number', 'unknown')
        
        for info in question.get('all_information', []):
            siblings = info.get('siblings', [])
            if siblings:
                sibling_count = len(siblings)
                sibling_counts.append(sibling_count)
                siblings_by_question[question_num].append(sibling_count)
    
    # Calculate statistics on sibling counts
    if sibling_counts:
        avg_siblings = sum(sibling_counts) / len(sibling_counts)
        max_siblings = max(sibling_counts)
        min_siblings = min(sibling_counts)
    else:
        avg_siblings = max_siblings = min_siblings = 0
    
    return {
        'sibling_stats': {
            'avg_siblings': avg_siblings,
            'max_siblings': max_siblings,
            'min_siblings': min_siblings,
            'total_info_with_siblings': len(sibling_counts)
        },
        'siblings_by_question': dict(siblings_by_question)
    }

def analyze_ancestor_relationships(data):
    """
    Analyze ancestor relationships in the gathered information
    """
    questions = data.get('questions', [])
    
    # Track depth of ancestry
    ancestry_depths = []
    
    # Track sections by frequency
    section_counts = Counter()
    
    for question in questions:
        for info in question.get('all_information', []):
            ancestors = info.get('ancestors', [])
            if ancestors:
                ancestry_depths.append(len(ancestors))
                
            section_id = info.get('section_id', 'None')
            if section_id and section_id != 'None':
                section_counts[section_id] += 1
    
    return {
        'ancestry_stats': {
            'avg_depth': sum(ancestry_depths) / len(ancestry_depths) if ancestry_depths else 0,
            'max_depth': max(ancestry_depths) if ancestry_depths else 0,
            'min_depth': min(ancestry_depths) if ancestry_depths else 0
        },
        'top_sections': dict(section_counts.most_common(15))
    }

def analyze_source_distribution_by_question(data):
    """
    Analyze how sources (table, figure, answer) are distributed across questions
    """
    questions = data.get('questions', [])
    
    source_distribution = []
    
    for question in questions:
        question_num = question.get('question_number', 'unknown')
        question_text = question.get('question', '')
        
        count_info = question.get('count_info', {})
        
        source_distribution.append({
            'question_num': question_num,
            'question': question_text,
            'table_count': count_info.get('table_pieces', 0),
            'figure_count': count_info.get('figure_pieces', 0),
            'answer_count': count_info.get('answer_pieces', 0),
            'total_count': count_info.get('total_pieces', 0)
        })
    
    return source_distribution

def create_report(data, output_file):
    """
    Create a comprehensive report of the analysis
    """
    # Generate basic stats
    basic_stats = generate_basic_stats(data)
    
    # Analyze parent-child relationships
    parent_child_analysis = analyze_parent_child_relationships(data)
    
    # Analyze sibling relationships
    sibling_analysis = analyze_sibling_relationships(data)
    
    # Analyze ancestor relationships
    ancestor_analysis = analyze_ancestor_relationships(data)
    
    # Analyze source distribution
    source_distribution = analyze_source_distribution_by_question(data)
    
    # Create a DataFrame for source distribution
    df_source = pd.DataFrame(source_distribution)
    df_source = df_source.sort_values(by='total_count', ascending=False)
    
    # Start building the report
    report = ["# Analysis of Gathered Information\n"]
    
    # Basic stats section
    report.append("## Basic Statistics\n")
    report.append(f"- Total questions: {basic_stats['total_questions']}")
    report.append(f"- Total information pieces: {basic_stats['total_information_pieces']}")
    report.append(f"- Average information pieces per question: {basic_stats['avg_info_per_question']:.2f}")
    report.append(f"- Questions with no information: {basic_stats['questions_with_no_info']}")
    
    # Source distribution
    report.append("\n### Source Distribution\n")
    for source, count in basic_stats['source_distribution'].items():
        report.append(f"- {source.capitalize()}: {count} pieces")
    
    # Questions with most information
    report.append("\n## Questions with Most Information\n")
    report.append("| Question # | Question | Table Count | Figure Count | Answer Count | Total Count |")
    report.append("|------------|----------|-------------|--------------|--------------|-------------|")
    
    for _, row in df_source.head(10).iterrows():
        question_text = row['question'][:50] + "..." if len(row['question']) > 50 else row['question']
        report.append(f"| {row['question_num']} | {question_text} | {row['table_count']} | {row['figure_count']} | {row['answer_count']} | {row['total_count']} |")
    
    # Questions with no information
    report.append("\n## Questions with No Information\n")
    no_info_questions = data['summary'].get('questions_with_no_information', [])
    if no_info_questions:
        for question_num in no_info_questions:
            question_text = next((q['question'] for q in data['questions'] if q['question_number'] == question_num), 'Unknown')
            report.append(f"- Question {question_num}: {question_text}")
    else:
        report.append("All questions have information associated with them.")
    
    # Sibling analysis
    report.append("\n## Sibling Analysis\n")
    report.append(f"- Average siblings per node: {sibling_analysis['sibling_stats']['avg_siblings']:.2f}")
    report.append(f"- Maximum siblings: {sibling_analysis['sibling_stats']['max_siblings']}")
    report.append(f"- Minimum siblings: {sibling_analysis['sibling_stats']['min_siblings']}")
    report.append(f"- Total information pieces with siblings: {sibling_analysis['sibling_stats']['total_info_with_siblings']}")
    
    # Questions with most siblings
    report.append("\n### Questions with Most Siblings\n")
    questions_by_avg_siblings = {}
    for question_num, sibling_counts in sibling_analysis['siblings_by_question'].items():
        questions_by_avg_siblings[question_num] = sum(sibling_counts) / len(sibling_counts)
    
    sorted_questions = sorted(questions_by_avg_siblings.items(), key=lambda x: x[1], reverse=True)
    
    report.append("| Question # | Avg Siblings |")
    report.append("|------------|--------------|")
    for question_num, avg_siblings in sorted_questions[:10]:
        report.append(f"| {question_num} | {avg_siblings:.2f} |")
    
    # Ancestor analysis
    report.append("\n## Ancestry Analysis\n")
    report.append(f"- Average depth of ancestors: {ancestor_analysis['ancestry_stats']['avg_depth']:.2f}")
    report.append(f"- Maximum ancestry depth: {ancestor_analysis['ancestry_stats']['max_depth']}")
    report.append(f"- Minimum ancestry depth: {ancestor_analysis['ancestry_stats']['min_depth']}")
    
    # Top sections
    report.append("\n### Most Common Sections\n")
    report.append("| Section ID | Count |")
    report.append("|------------|-------|")
    for section, count in ancestor_analysis['top_sections'].items():
        section_display = section[:50] + "..." if len(section) > 50 else section
        report.append(f"| {section_display} | {count} |")
    
    # Common parent-child relationships
    report.append("\n## Most Common Parent-Child Relationships\n")
    report.append("| Relationship | Count |")
    report.append("|--------------|-------|")
    for relationship, count in parent_child_analysis['parent_child_counts'].items():
        report.append(f"| {relationship} | {count} |")
    
    # Save the report
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    
    return '\n'.join(report)

def generate_visualizations(data, output_dir):
    """
    Generate visualizations of the analysis
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Source distribution pie chart
    plt.figure(figsize=(10, 6))
    source_dist = data['summary']['source_distribution']
    plt.pie(source_dist.values(), labels=source_dist.keys(), autopct='%1.1f%%')
    plt.title('Information Source Distribution')
    plt.savefig(os.path.join(output_dir, 'source_distribution.png'))
    plt.close()
    
    # Information count by question
    source_distribution = analyze_source_distribution_by_question(data)
    df_source = pd.DataFrame(source_distribution)
    df_source = df_source.sort_values(by='question_num')
    
    plt.figure(figsize=(15, 8))
    
    # Create stacked bar chart
    x = range(len(df_source))
    plt.bar(x, df_source['table_count'], label='Table')
    plt.bar(x, df_source['figure_count'], bottom=df_source['table_count'], label='Figure')
    plt.bar(x, df_source['answer_count'], 
            bottom=df_source['table_count'] + df_source['figure_count'], 
            label='Answer')
    
    plt.xlabel('Question Number')
    plt.ylabel('Information Count')
    plt.title('Information Count by Question and Source Type')
    plt.xticks(x, df_source['question_num'], rotation=90)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'info_by_question.png'))
    plt.close()
    
    # Sibling count distribution
    sibling_analysis = analyze_sibling_relationships(data)
    sibling_counts = []
    for counts in sibling_analysis['siblings_by_question'].values():
        sibling_counts.extend(counts)
    
    if sibling_counts:
        plt.figure(figsize=(10, 6))
        plt.hist(sibling_counts, bins=max(10, min(50, max(sibling_counts))), alpha=0.7)
        plt.xlabel('Number of Siblings')
        plt.ylabel('Frequency')
        plt.title('Distribution of Sibling Counts')
        plt.savefig(os.path.join(output_dir, 'sibling_distribution.png'))
        plt.close()
    
    # Ancestry depth distribution
    ancestor_analysis = analyze_ancestor_relationships(data)
    questions = data.get('questions', [])
    ancestry_depths = []
    
    for question in questions:
        for info in question.get('all_information', []):
            ancestors = info.get('ancestors', [])
            if ancestors:
                ancestry_depths.append(len(ancestors))
    
    if ancestry_depths:
        plt.figure(figsize=(10, 6))
        plt.hist(ancestry_depths, bins=max(5, min(20, max(ancestry_depths)+1)), alpha=0.7)
        plt.xlabel('Ancestry Depth')
        plt.ylabel('Frequency')
        plt.title('Distribution of Ancestry Depths')
        plt.savefig(os.path.join(output_dir, 'ancestry_distribution.png'))
        plt.close()

def main():
    # File paths
    input_file = '/home/yeskey525/Research_CODE/experiments/golden_answers/gathered_information.json'
    report_file = '/home/yeskey525/Research_CODE/experiments/golden_answers/analysis_report.md'
    viz_dir = '/home/yeskey525/Research_CODE/experiments/golden_answers/visualizations'
    
    # Load the gathered information
    data = load_gathered_info(input_file)
    print(f"Loaded data from {input_file}")
    
    # Create the analysis report
    report = create_report(data, report_file)
    print(f"Analysis report saved to {report_file}")
    
    # Generate visualizations
    try:
        generate_visualizations(data, viz_dir)
        print(f"Visualizations saved to {viz_dir}")
    except Exception as e:
        print(f"Error generating visualizations: {e}")
    
    print("Analysis complete!")

if __name__ == "__main__":
    main() 