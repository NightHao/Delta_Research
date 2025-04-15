from query_processor import QueryProcessor
from asyncio import run
from langchain_openai import ChatOpenAI
import re
from dotenv import load_dotenv, find_dotenv
import json

async def main():
    load_dotenv()
    llm = ChatOpenAI(model="o1", reasoning_effort="medium")
    # Load questions and prompts from final_prompt.json
    try:
        with open("final_prompt.json", "r") as f:
            prompt_dict = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        print("Error: final_prompt.json not found or invalid")
        prompt_dict = {}
    
    # Try to load existing answers from o1_answers.json
    try:
        with open("o1_answers.json", "r") as f:
            answers = json.load(f)
        print(f"Loaded {len(answers)} existing answers from o1_answers.json")
    except (FileNotFoundError, json.JSONDecodeError):
        print("No existing answers found or invalid file. Starting fresh.")
        answers = {}
    
    # Iterate through each question and its prompt
    for question, prompt in prompt_dict.items():
        if question in answers:
            print(f"Skipping question: {question} (already answered)")
            continue
        print(f"Processing question: {question}")
        
        # Invoke the LLM with the prompt
        response = llm.invoke(prompt)
        final_answer = response.content
        
        # Format the answer
        final_answer = re.sub(
            r'(\d+\. \*\*[^:]+\*\*): ', 
            r'\n### \1\n', 
            final_answer
        )
        
        # Store the answer
        answers[question] = final_answer
        
        # Print the answer
        print(f"\nAnswer:\n{final_answer}")
        print("\n" + "-"*50 + "\n")
        
        # Save answers after each response
        with open("o1_answers.json", "w") as f:
            json.dump(answers, f, indent=4)
        print(f"Updated answers saved to o1_answers.json")

    # Final message
    print(f"Completed processing {len(answers)} questions. All answers saved to o1_answers.json")

def format_text_to_string(input_text):
    """
    Converts any input text into a properly formatted string.
    
    Args:
        input_text (str): The text to be formatted
        
    Returns:
        str: The formatted string
    """
    # Remove any extra whitespace and ensure proper formatting
    return input_text.strip()

def get_user_input_as_string():
    """
    Gets multi-line input from the user and formats it as a string.
    User can end input by typing 'END' on a new line.
    
    Returns:
        str: The formatted user input as a string
    """
    print("Enter your text (type 'END' on a new line when finished):")
    lines = []
    while True:
        line = input()
        if line.strip() == "END":
            break
        lines.append(line)
    
    # Join the lines and format the text
    return format_text_to_string("\n".join(lines))

def save_text_to_json(text, filename="formatted_text.json", key="text"):
    """
    Saves the given text to a JSON file.
    
    Args:
        text (str): The text to save
        filename (str): The name of the JSON file
        key (str): The key to use in the JSON object
    
    Returns:
        str: Path to the saved file
    """
    data = {key: text}
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    return filename

if __name__ == "__main__":
    run(main())
    
    # Get input from user and format it
    # formatted_text = get_user_input_as_string()
    # print("\nFormatted text:")
    # print(formatted_text)
    
    # # Save to JSON file
    # output_file = save_text_to_json(formatted_text)
    # print(f"\nText saved to {output_file}")
    