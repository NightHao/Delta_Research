import csv
import json

# Read from the CSV file
results = {}
with open('graphrag_ans.csv', 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        question = row['question']
        answer = row['generated_answer']
        results[question] = answer

# Write to JSON file
with open('graphrag_ans.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=4, ensure_ascii=False)

print("Conversion completed successfully!")
