from prompt_eng import process_single_file

# Process the EIM file and get the new prompt
file_path = "/home/yeskey525/Research_CODE/merged_prompts/What is EIM?.txt"
new_prompt = process_single_file(file_path)

# Save to a file we can examine
with open("/home/yeskey525/Research_CODE/sample_processed_eim.txt", "w", encoding="utf-8") as f:
    f.write(new_prompt)

print("Processed prompt saved to /home/yeskey525/Research_CODE/sample_processed_eim.txt") 