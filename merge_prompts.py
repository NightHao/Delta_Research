import os
import glob
import re
import json

# Path to the complete_prompts directory
complete_prompts_dir = '/home/yeskey525/Research_CODE/experiments/golden_answers_exe/complete_prompts'

# Path to the final_prompt.json file
final_prompt_file = '/home/yeskey525/Research_CODE/final_prompt.json'

# Load the final prompt content
with open(final_prompt_file, 'r', encoding='utf-8') as f:
    final_prompt_data = json.load(f)

# Dictionary to map simplified questions to their formal versions
replace_dict = {
    "What is TSS?": "What is TEST SUITE STRUCTURE?",
    "What is ATS?": "What is ABSTRACT TEST SUITE?",
    "What is EIM?": "What is External Identification Means?",
    "What is HPGP?": "What is HOMEPLUG GREEN PHY?",
    "What is CCo?": "What is Central Coordinator?",
    "What is SLAC?": "What is SIGNAL LEVEL ATTENUATION CHARACTERIZATION?",
    "What is the difference between MTC and PTC?": "What is the difference between Main Test Component and Parallel Test Component?",
    "What is the difference between Data SAP and Data link control SAP?": "What is the difference between Service Access Point for Data and Data link control SAP?",
}

# Function to extract question from prompt file name
def extract_question(filename):
    # Get the base name without directory and extension
    base = os.path.basename(filename)
    base = os.path.splitext(base)[0]
    
    # For special protocol commands like CM_SLAC_PARM.REQ or D-LINK_PAUSE.request
    # We need to extract the full command before the first underscore after the command
    if 'CM_' in base or 'D-LINK_' in base:
        # Check if there's a pattern like "What is X?" where X contains CM_ or D-LINK_
        match = re.match(r'What is (CM_[A-Z_.]+|D-LINK_[A-Za-z_.]+)\??', base)
        if match:
            protocol_command = match.group(1)
            return f"What is {protocol_command}?"
    
    # For regular questions, extract everything before first underscore or the entire string
    if '_' in base:
        question_part = base.split('_', 1)[0]
    else:
        question_part = base
    
    # Make sure the question ends with a question mark
    if not question_part.endswith('?'):
        question_part += '?'
    
    return question_part

# Process each file in the complete_prompts directory
prompt_files = glob.glob(os.path.join(complete_prompts_dir, '*.txt'))

# Create a directory to store merged prompts
output_dir = '/home/yeskey525/Research_CODE/merged_prompts'
os.makedirs(output_dir, exist_ok=True)

# Process each file
for prompt_file in prompt_files:
    # Get the question from the filename
    question = extract_question(prompt_file)
    
    # Check if the question needs to be replaced with a formal version
    formal_question = replace_dict.get(question, question)
    
    # Read the content of the complete prompt file
    with open(prompt_file, 'r', encoding='utf-8') as f:
        complete_prompt_content = f.read()
    
    # For debugging
    print(f"File: {os.path.basename(prompt_file)}")
    print(f"Extracted question: {question}")
    if question != formal_question:
        print(f"Replaced with formal question: {formal_question}")
    
    # Look for the question in the final_prompt_data keys
    found = False
    
    # The keys in final_prompt_data might not have a question mark
    # so we'll try different variations
    variations = [
        formal_question,                      # Formal question (with question mark)
        formal_question.replace('?', ''),     # Without question mark
        formal_question.strip()               # Trimmed spaces
    ]
    
    for q_variation in variations:
        if q_variation in final_prompt_data:
            final_prompt_section = final_prompt_data[q_variation]
            
            # Remove the last sentence about answering the question
            final_prompt_section = re.sub(
                r'You need to answer the following question as more details as possible based on the provided information above\n Question: .*?$', 
                '', 
                final_prompt_section,
                flags=re.DOTALL
            )
            
            # Remove the redundant question format at the beginning of the complete_prompt
            # Original format: "Question What is X?: What is X?"
            redundant_question_pattern = rf"^Question {re.escape(question)}: {re.escape(question)}\n+"
            modified_complete_prompt = re.sub(redundant_question_pattern, "", complete_prompt_content, flags=re.MULTILINE)
            
            # Extract the "Below is all..." paragraph
            info_paragraph_pattern = r"(Below is all the available information.*?Do not add any new information that is not present below\.)\n+"
            info_paragraph_match = re.search(info_paragraph_pattern, modified_complete_prompt, re.DOTALL)
            
            if info_paragraph_match:
                info_paragraph = info_paragraph_match.group(1)
                # Change "Below" to "Above"
                info_paragraph = info_paragraph.replace("Below is", "Above is")
                # Remove the paragraph from its original position
                modified_complete_prompt = re.sub(info_paragraph_pattern, "", modified_complete_prompt, flags=re.DOTALL)
            else:
                info_paragraph = ""
            
            # Merge with final_prompt content FIRST, then modified complete_prompt content
            # Finally add the question and modified info paragraph at the bottom
            merged_content = final_prompt_section + '\n\n' + modified_complete_prompt + '\n\n' + f"Question: {question}\n\n" + info_paragraph
            
            # Write the merged content to a new file
            output_file = os.path.join(output_dir, os.path.basename(prompt_file))
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(merged_content)
            
            print(f"Processed: {os.path.basename(prompt_file)}")
            found = True
            break
    
    if not found:
        print(f"Warning: Could not find matching section for {formal_question} in final prompt")

print("Merging complete!") 