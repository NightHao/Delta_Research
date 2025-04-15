import json
import codecs

class AliasBuilder:
    def __init__(self):
        self.log_file_path = './log.json'
        self.output_path = './alias_dict.json'

    def load_abbr_dict(self):
        """Load the abbreviation dictionary from a JSON file."""
        with codecs.open(self.log_file_path, 'r', encoding='utf-8') as f:
            log_data = json.load(f)
        return log_data.get('abbr_dict', {}).get('data', {})

    def normalize_full_name(self, full_name):
        """Normalize a full name with proper capitalization of words and hyphenated parts."""
        words = []
        
        # Split by spaces and process each word
        for word in full_name.strip().split():
            if '-' in word:
                # Handle hyphenated words by capitalizing each part
                hyphenated_parts = [part.capitalize() for part in word.split('-')]
                words.append('-'.join(hyphenated_parts))
            else:
                # Regular word capitalization
                words.append(word.capitalize())
        
        return ' '.join(words)
    
    def build_alias_dict(self, abbr_dict):
        """
        Build a bidirectional alias dictionary from the abbreviation dictionary.
        
        The resulting dictionary will have two keys:
        - 'abbreviations': Maps abbreviations to lists of full names
        - 'full_names': Maps full names to lists of abbreviations
        """
        alias_dict = {
            'abbreviations': abbr_dict,
            'full_names': {}
        }
        
        # Build the full_name to abbreviation mapping
        for abbr, full_names in abbr_dict.items():
            for full_name in full_names:
                if full_name:
                    normalized_full_name = self.normalize_full_name(full_name)
                    if normalized_full_name not in alias_dict['full_names']:
                        alias_dict['full_names'][normalized_full_name] = []
                    
                    if abbr not in alias_dict['full_names'][normalized_full_name]:
                        alias_dict['full_names'][normalized_full_name].append(abbr)
        
        return alias_dict

    def save_alias_dict(self, alias_dict):
        """Save the alias dictionary to a JSON file."""
        with codecs.open(self.output_path, 'w', encoding='utf-8') as f:
            json.dump(alias_dict, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    alias_builder = AliasBuilder()
    abbr_dict = alias_builder.load_abbr_dict()
    alias_dict = alias_builder.build_alias_dict(abbr_dict)
    alias_builder.save_alias_dict(alias_dict)
    
    print(f"Alias dictionary created and saved to ./alias_dict.json")
    print(f"Number of abbreviations: {len(alias_dict['abbreviations'])}")
    print(f"Number of full names: {len(alias_dict['full_names'])}")