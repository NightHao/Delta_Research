#!/usr/bin/env python3
import os, json, glob, re
from langchain_openai import AzureChatOpenAI
from langchain.schema import HumanMessage
from langchain.callbacks.tracers.langchain import wait_for_all_tracers
from dotenv import load_dotenv

load_dotenv()

# Paths and setup
PROMPTS_DIR = '/home/yeskey525/Research_CODE/experiments/golden_answers/complete_prompts'
RESULTS_FILE = '/home/yeskey525/Research_CODE/experiments/golden_answers/results/golden_answers.json'
os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)

# Extract the base endpoint and api version from the full URL
full_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
base_endpoint = re.match(r'(https://.*?)/openai/deployments', full_endpoint).group(1)
api_version_match = re.search(r'api-version=([^&]+)', full_endpoint)
api_version = api_version_match.group(1) if api_version_match else "2023-05-15"

# Initialize Azure OpenAI with environment variables
llm = AzureChatOpenAI(
    model="o4-mini",
    azure_endpoint=base_endpoint,
    api_version=api_version
)
try:
    llm.invoke("Hello, World!")
finally:
    wait_for_all_tracers()
# Process all prompts and save results
results = {}
for prompt_file in glob.glob(os.path.join(PROMPTS_DIR, "*.txt")):
    with open(prompt_file, 'r', encoding='utf-8') as f:
        prompt_content = f.read()
    # Extract just the question part from the first line
    # Format is typically "Question What is X?: What is X?"
    # We only want the second part "What is X?"
    first_line = prompt_content.split('\n')[0]
    if ': ' in first_line:
        question = first_line.split(': ')[1].strip()
    else:
        # Fallback if the expected format is not found
        question = first_line.strip('?').strip() + '?'
    print(f"Processing: {question}")
    response = llm.invoke([HumanMessage(content=prompt_content)])
    results[question] = response.content.strip()
    with open(RESULTS_FILE, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

print(f"Completed! Processed {len(results)} prompts. Results saved to {RESULTS_FILE}") 