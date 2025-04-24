import json

# Load the original alias dictionary
with open('alias_dict.json', 'r', encoding='utf-8') as f:
    alias_dict = json.load(f)

# Function to replace spaces with underscores
def replace_spaces(text):
    return text.replace(' ', '_')

# Process abbreviations section
new_abbreviations = {}
for abbr, expansions in alias_dict['abbreviations'].items():
    new_expansions = [replace_spaces(exp) for exp in expansions]
    new_abbreviations[abbr] = new_expansions

# Process full_names section
new_full_names = {}
for full_name, abbrs in alias_dict['full_names'].items():
    new_full_name = replace_spaces(full_name)
    new_full_names[new_full_name] = abbrs

# Create new dictionary
new_alias_dict = {
    'abbreviations': new_abbreviations,
    'full_names': new_full_names
}

# Save the new dictionary
with open('alias_dict_underscored.json', 'w', encoding='utf-8') as f:
    json.dump(new_alias_dict, f, indent=4, ensure_ascii=False)

print("Created new alias dictionary with underscores instead of spaces.")
