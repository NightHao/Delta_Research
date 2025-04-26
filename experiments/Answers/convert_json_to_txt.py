import json
import os

def convert_json_to_formatted_txt(json_file_path):
    # Get the base name without extension
    base_name = os.path.splitext(json_file_path)[0]
    txt_file_path = base_name + '.txt'
    
    try:
        # Read JSON file
        with open(json_file_path, 'r', encoding='utf-8') as json_file:
            data = json.load(json_file)
        
        # Write to TXT file
        with open(txt_file_path, 'w', encoding='utf-8') as txt_file:
            if isinstance(data, dict):
                for question, answer in data.items():
                    # Write question
                    txt_file.write(f"Question: {question}\n\n")
                    # Write answer - the \n in the string will be automatically converted to newlines
                    txt_file.write(f"Answer: {answer}\n")
                    txt_file.write("\n" + "="*80 + "\n\n")  # Separator between QA pairs
            else:
                # If it's not a dict, just write the content directly
                txt_file.write(str(data))
                
        print(f"Successfully converted {json_file_path} to {txt_file_path}")
        
    except FileNotFoundError:
        print(f"Error: File {json_file_path} not found")
    except json.JSONDecodeError:
        print(f"Error: {json_file_path} is not a valid JSON file")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    json_files = [
        "./o4mini_all_info_ans.json"
    ]
    
    for json_file in json_files:
        if os.path.exists(json_file):
            convert_json_to_formatted_txt(json_file)