import json
import re

def convert_txt_to_json(input_file, output_file):
    """
    Convert a text file with questions and answers to JSON format.
    Questions become keys, answers become values.
    """
    with open(input_file, 'r', encoding='utf-8') as file:
        content = file.read()
    
    # Use regex to find all question-answer pairs
    pattern = r'Question: (.*?)\n\nAnswer: (.*?)(?=\n={10,}|\Z)'
    matches = re.findall(pattern, content, re.DOTALL)
    
    # Dictionary to store question-answer pairs
    qa_dict = {}
    
    for question, answer in matches:
        qa_dict[question.strip()] = answer.strip()
    
    # Write to JSON file
    with open(output_file, 'w', encoding='utf-8') as file:
        json.dump(qa_dict, file, indent=4, ensure_ascii=False)
    
    print(f"Converted {len(qa_dict)} question-answer pairs to {output_file}")

if __name__ == "__main__":
    input_file = "/home/yeskey525/Research_CODE/experiments/golden_answers/ans/o1_answers.txt"
    output_file = "/home/yeskey525/Research_CODE/experiments/golden_answers/ans/o1_answers.json"
    convert_txt_to_json(input_file, output_file)
