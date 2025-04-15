from openai import OpenAI, AsyncOpenAI
from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI
from langchain.callbacks.tracers.langchain import wait_for_all_tracers
import os, re, json, asyncio, nest_asyncio, time, numpy as np, pprint, datetime
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
from pydantic import BaseModel, Field

class EntityGraphProcessor:
    def __init__(self):
        # Find the .env file
        env_path = find_dotenv()
        print(f".env file found at: {env_path}")
        load_dotenv()
        print("Environment variables loaded.")
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
        try:
            self.llm.invoke("Hello, World!")
        finally:
            wait_for_all_tracers()

    def display_env_variables(self, exclude_keys=None):
        """
        Displays environment variables, excluding or masking sensitive keys.

        Args:
            exclude_keys (list, optional): List of keys to exclude or mask. Defaults to None.
        """
        if exclude_keys is None:
            exclude_keys = []
        
        print("Loaded Environment Variables:")
        for key, value in os.environ.items():
            if key in exclude_keys:
                print(f"{key}=****")  # Mask sensitive values
            elif key.upper().endswith("_KEY") or key.upper().startswith("SECRET"):
                print(f"{key}=****")  # Additional masking based on naming conventions
            else:
                print(f"{key}={value}")

    def load_log(self, file_path: str = 'log.json') -> dict:
        """
        Loads the log from log.json. Creates empty log if file doesn't exist.
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            return {}
        except json.JSONDecodeError:
            return {}

    def write_to_log(self, data, step_name: str, file_path: str = 'log.json') -> None:
        """
        Writes processing outputs to log.json with timestamps.
        
        Args:
            data: The output data to store
            step_name: Name of the processing step (e.g., 'abbreviation_results', 'pattern_clusters')
            file_path: Path to the outputs file
        """
        # Load existing outputs
        outputs = self.load_log(file_path)
        
        # Add metadata to the output
        output_entry = {
            'data': data,
            'timestamp': datetime.datetime.now().isoformat(),
            'step': step_name
        }
        
        # Update with new data
        outputs[step_name] = output_entry
        
        # Write back to file
        with open(file_path, 'w', encoding='utf-8') as outfile:
            json.dump(outputs, outfile, indent=4, ensure_ascii=False)

    def clear_log(self, file_path: str = 'log.json') -> None:
        """
        Clears the log by writing an empty dictionary to the file.
        """
        with open(file_path, 'w', encoding='utf-8') as outfile:
            json.dump({}, outfile, indent=4, ensure_ascii=False)

    def load_entity_graph(self, file_path: str) -> dict:
        """
        Loads the entity graph from a JSON file.
        Returns a dictionary where keys are node names and each value is its data.
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading entity graph: {e}")
            return {}


    def save_graph(self, file_path: str, data) -> None:
        """
        Save graph as a JSON file.
        """
        with open(file_path, 'w', encoding='utf-8') as outfile:
            json.dump(data, outfile, indent=4, ensure_ascii=False)

    def extract_abbreviation_info(self, description: str, node_name: str) -> list[tuple[str, str]]:
        """
        Extracts abbreviations and their full names from a multi-line description.
        Returns a list of tuples [(abbr, full), ...] for all valid matches found.
        """
        if not description or '_' in node_name:
            return []

        pairs = []
        # Process each line independently
        lines = [line.strip() for line in description.split('\n') if line.strip()]
        regex_patterns = self.load_log()['regex_patterns']['data']
        for line in lines:
            for pattern in regex_patterns:
                pattern = re.compile(pattern)
                match = pattern.search(line)
                if match:
                    if(len(match.group(1).strip()) >= len(match.group(2).strip())):
                        full = match.group(1).strip()
                        abbr = match.group(2).strip()
                    else:
                        full = match.group(2).strip()
                        abbr = match.group(1).strip()
                    words = full.split()
                    if len(words) <= len(abbr):
                        pairs.append((abbr, full))
                        continue
                    
        return list(set(pairs))

    def normalize_text(self, name: str) -> str:
        """
        Returns a normalized lookup key by lowercasing and trimming spaces.
        """
        return "".join(name.strip().lower().split())

    def merge_nodes(self, parent: dict, child: dict) -> None:
        """
        Merges the data from the child node into the parent node.
        (Merging types, descriptions, and connections.)
        """
        # Merge types: if different, append with a slash.
        if child.get("type"):
            if parent.get("type"):
                if child["type"] not in parent["type"]:
                    parent["type"] = parent["type"] + "/" + child["type"]
            else:
                parent["type"] = child["type"]

        # Merge descriptions: union of non-empty lines.
        parent_lines = set(line.strip() for line in parent.get("description", "").split("\n") if line.strip())
        child_lines  = set(line.strip() for line in child.get("description", "").split("\n") if line.strip())
        merged = parent_lines.union(child_lines)
        parent["description"] = "\n".join(sorted(merged))

        # Merge connections (concatenation; deduplication can be added if needed).
        parent_conns = parent.get("connections", [])
        child_conns  = child.get("connections", [])
        parent["connections"] = parent_conns + child_conns

    def update_connection_targets(self, entity_graph: dict, old_key: str, new_key: str) -> None:
        """
        Updates all connection targets in the graph that point to old_key so they point to new_key.
        """
        for node, data in entity_graph.items():
            for conn in data.get("connections", []):
                if conn.get("target") == old_key:
                    conn["target"] = new_key

    def remove_self_loops(self, entity_graph: dict) -> None:
        """
        For every node in the entity graph, remove any connection whose target is the same as the node key.
        """
        for node_key, data in entity_graph.items():
            if "connections" in data:
                data["connections"] = [conn for conn in data["connections"] if conn.get("target") != node_key]

    def merge_duplicate_nodes(self, entity_graph: dict) -> dict:
        """
        Merges nodes that have the same name (case-insensitive).
        Returns the merged graph.
        """
        merged = {}  # maps normalized node names to the canonical key
        keys_to_remove = set()

        for key in list(entity_graph.keys()):
            key_lc = self.normalize_text(key)
            if key_lc in merged:
                canonical_key = merged[key_lc]
                self.merge_nodes(entity_graph[canonical_key], entity_graph[key])
                self.update_connection_targets(entity_graph, key, canonical_key)
                keys_to_remove.add(key)
            else:
                merged[key_lc] = key

        for key in keys_to_remove:
            if key in entity_graph:
                del entity_graph[key]
        
        return entity_graph

    def merge_abbreviation_nodes(self, entity_graph: dict, abbr_dict: dict) -> None:
        """
        Merge abbreviation nodes into their corresponding full name nodes.
        Always keeps the full name node when available.
        
        Args:
            entity_graph: The graph to process
            abbr_dict: Bidirectional mapping between abbreviations and full names

        Returns:
            Processed graph with merged nodes
        """
        for abbrev, full_names in abbr_dict.items():
            # Skip if abbreviation node doesn't exist
            if abbrev not in entity_graph:
                continue
                
            # Find matching full name nodes
            full_name_nodes = [name.upper() for name in full_names if name.upper() in entity_graph]
            
            if full_name_nodes:
                # Choose the first full name node as target
                target_node = full_name_nodes[0]
                
                # Merge abbreviation node into full name node
                self.merge_nodes(entity_graph[target_node], entity_graph[abbrev])
                self.update_connection_targets(entity_graph, abbrev, target_node)
                del entity_graph[abbrev]

    async def extract_abbreviations(self, node_names: list[str]) -> list[str]:
        """
        Extract abbreviations from a list of node names using GPT-4o-mini.
        
        Args:
            node_names: List of node names to analyze
             
        Returns:
            List of identified abbreviations
        """

        input_set = set([name.strip() for name in node_names])
        node_names_str = ""
        for name in input_set:
            node_names_str += f'{name}\n'
        gpt_4o_mini = ChatOpenAI(model="gpt-4o-mini", temperature=0.0, seed = 42)
        json_llm = gpt_4o_mini.bind(response_format={'type': 'json_object'})
        ai_msg = json_llm.invoke(f"""Return a JSON object containing only the abbreviations that exactly appear in the provided node names. An abbreviation is defined as a short, all-uppercase string that represents a longer full name. In our domain, abbreviations are typically less than 7 characters and may include hyphens. Do not include longer descriptive names or identifiers with mixed case.
                                The node names are: {node_names_str}""")
        print(ai_msg)
        try:
            # Parse the JSON response from the content field
            response_dict = json.loads(ai_msg.content)
            # Extract the abbreviations list
            abbrevs = response_dict.get('abbreviations', [])
            # print(abbrevs)
            # Filter to only include abbreviations that exist in the input set
            valid_abbrevs = [abbr for abbr in abbrevs if abbr in input_set]
            return valid_abbrevs
        except json.JSONDecodeError:
            print("Error parsing JSON response")
            return []
        except Exception as e:
            print(f"Unexpected error: {e}")
            return []

    def escape_literal(self, literal):
        """
        Escape a literal string for use in a regex, and replace any whitespace 
        with \s+ so that multiple spaces/tabs can match.
        """
        # Split literal into segments (keeping whitespace groups)
        parts = re.split(r'(\s+)', literal)
        escaped_parts = []
        for part in parts:
            if part.isspace():
                # Replace any whitespace group with \s+
                escaped_parts.append(r"\s+")
            else:
                # Escape other characters literally
                escaped_parts.append(re.escape(part))
        return ''.join(escaped_parts)

    def template_to_regex(self, template):
        """
        Convert a template (e.g., "[ABBR] is an abbreviation for [FULL_NAME]") into a regex.
        Placeholders in the template are replaced with corresponding regex patterns.
        """
        PLACEHOLDER_MAP = {
        "[ABBR]": r"(?P<ABBR>[A-Z][A-Za-z0-9-]*(?:\s+[A-Z][A-Za-z0-9-]*)*\.?)",
        "[FULL_NAME]": r"(?P<FULL_NAME>(?:(?:the|a|an)\s+)?[A-Z][a-zA-Z0-9-]+(?:\s+[A-Z][a-zA-Z0-9-]+)+)"
        }

        # This pattern will find any placeholder that is exactly [ABBR] or [FULL_NAME].
        placeholder_pattern = re.compile(r"(\[ABBR\]|\[FULL_NAME\])")
        # Split the template into tokens (literals and placeholders)
        tokens = placeholder_pattern.split(template)
        
        regex_parts = []
        for token in tokens:
            if token in PLACEHOLDER_MAP:
                # Replace the placeholder with its regex group.
                regex_parts.append(PLACEHOLDER_MAP[token])
            elif token:
                # Process literal text (normalize whitespace)
                regex_parts.append(self.escape_literal(token))
        
        # Join the parts into a full regex pattern.
        # We add word-boundary markers if desired.
        full_pattern = r"\b" + ''.join(regex_parts) + r"\b"
        return full_pattern

    async def extract_pattern(self, abbrev_to_descriptions: dict[str, str]) -> dict[str, str]:
        """
        Extract patterns that directly define abbreviations in the descriptions 
        and identify the full names for each abbreviation.
        """
        o3_mini = ChatOpenAI(model="o3-mini", reasoning_effort = 'low', seed = 42)
        json_llm = o3_mini.bind(response_format={'type': 'json_object'})
        abbrev_to_descriptions_str = '<abbreviation_entries>\n'
        for abbr, desc in abbrev_to_descriptions.items():
            abbrev_to_descriptions_str += '<entry>\n'
            abbrev_to_descriptions_str += f'<abbreviation>{abbr}</abbreviation>\n'
            abbrev_to_descriptions_str += f'<description>{desc}</description>\n'
            abbrev_to_descriptions_str += '</entry>\n'
        abbrev_to_descriptions_str += '</abbreviation_entries>'
        ai_msg = json_llm.invoke(f"""You are tasked with analyzing a text containing abbreviations and their descriptions. Your objectives are:
    1. Find ALL complete full names for each abbreviation from its description
    2. Extract the specific patterns used to define those abbreviations

    Here is the text to analyze:
    {abbrev_to_descriptions_str}

    Follow these rules when identifying full names and patterns:
    1. For each abbreviation, identify ALL its EXACT full names as stated in the description
    2. Extract full names that are EXPLICITLY defined as equivalent to the abbreviation OR appear at the beginning of the description as an implied definition
    3. If multiple distinct full names exist for an abbreviation, include ALL of them in a list
    4. For patterns, ONLY include those showing a direct equivalence (use "[ABBR]" and "[FULL_NAME]" placeholders in your pattern descriptions)

    Valid full name definition patterns include:
    - "[ABBR] stands for [FULL_NAME]"
    - "[ABBR], which means [FULL_NAME]"
    - "[ABBR] (short for [FULL_NAME])"
    - "[ABBR] is an abbreviation for [FULL_NAME]"
    - "[ABBR] refers to [FULL_NAME]"
    - "[FULL_NAME] ([ABBR])"
    - "[ABBR] ([FULL_NAME])"

    Invalid patterns (these show relationships but not direct equivalence):
    - "[ABBR] is a protocol that allows [FULL_NAME]"
    - "[ABBR] is a standard document from [FULL_NAME]"
    - "[FULL_NAME] is a conceptual framework used to understand [ABBR]"

    To process the input:
    1. Read through the text carefully
    2. Identify all abbreviations and their corresponding full names
    3. Note the patterns used to define each abbreviation
    4. Ensure that only valid patterns are included
    5. Create a list of all unique patterns found
    6. For each abbreviation, create a list of all its full names

    Your final output should be a JSON object with two main components:
    1. "patterns": an array of all unique direct equivalence patterns found
    2. "abbreviation_expansions": an object mapping each abbreviation to an array of its full names

    Example output format:
    <example_output>
    {{
    "patterns": ["[ABBR] ([FULL_NAME])", "[ABBR] stands for [FULL_NAME]"],
    "abbreviation_expansions": {{
        "EV": ["Electric Vehicle", "Extremely fast Vehicle"],
        "PnC": ["Plug and Charge"]
    }}
    }}
    </example_output>

    Ensure your final output contains only the JSON object as described above, without any additional explanation or commentary.""")    

        print(ai_msg)
        try:
            # Parse the JSON response from the content field
            response_dict = json.loads(ai_msg.content)
            patterns = response_dict.get('patterns', [])
            abbreviation_expansions = response_dict.get('abbreviation_expansions', {})
            
            # Convert patterns to regex
            regex_patterns = set()
            for pattern in patterns:
                # Replace placeholders with regex capture groups
                regex_patterns.add(self.template_to_regex(pattern))            
                
            # Return both the regex patterns and abbreviation expansions
            return {
                "regex_patterns": list(regex_patterns),
                "abbreviation_expansions": abbreviation_expansions
            }
        except json.JSONDecodeError:
            print("Error parsing JSON response")
            return {"regex_patterns": [], "abbreviation_expansions": {}}
        except Exception as e:
            print(f"Unexpected error: {e}")
            return {"regex_patterns": [], "abbreviation_expansions": {}}

    def cluster_patterns_by_similarity(self, pattern_list: list[str]) -> dict[int, list[str]]:
        pattern_clusters = {}
        # Handle empty or single pattern case
        if len(pattern_list) == 0:
            print("No patterns found, created empty pattern clusters.")
        elif len(pattern_list) == 1:
            pattern_clusters = {0: pattern_list}
            print("Only one pattern found, assigned to a single cluster.")
        else:
            client = OpenAI()
            # Preprocess patterns for better embedding
            processed_patterns = []
            for p in pattern_list:
                # Replace regex components with more readable text
                cleaned = p.replace('\\b', '')
                cleaned = cleaned.replace('(?P<ABBR>[A-Z][A-Za-z0-9-]*(?:\\s+[A-Z][A-Za-z0-9-]*)*\\.?)', 'ABBREVIATION')
                cleaned = cleaned.replace('(?P<FULL_NAME>(?:(?:the|a|an)\\s+)?[A-Z][a-zA-Z0-9-]+(?:\\s+[A-Z][a-zA-Z0-9-]+)+)', 'FULL NAME')
                processed_patterns.append(cleaned)
            
            print("Processed patterns for embedding:", processed_patterns)
            
            # Get embeddings from OpenAI
            response = client.embeddings.create(
                input=processed_patterns,
                model="text-embedding-3-small"
            )
            
            # Extract embeddings from response
            pattern_embeddings = np.array([data.embedding for data in response.data])
            
            # Calculate similarity matrix
            similarity_matrix = cosine_similarity(pattern_embeddings)
            
            # Due to floating point precision, ensure similarity is in [0,1]
            similarity_matrix = np.clip(similarity_matrix, 0, 1)
            
            # Print similarity matrix for debugging
            # print("Similarity matrix:")
            # for i in range(len(processed_patterns)):
            #     for j in range(len(processed_patterns)):
            #         print(f"{i} and {j}: {similarity_matrix[i, j]:.4f}")
            
            # Use directly a simpler algorithm: if similarity > threshold, put in same cluster
            similarity_threshold = 0.93
            
            # Start with each pattern in its own cluster
            n_patterns = len(pattern_list)
            cluster_assignments = list(range(n_patterns))
            
            # For each pair of patterns
            for i in range(n_patterns):
                for j in range(i+1, n_patterns):
                    # If similarity exceeds threshold, merge their clusters
                    if similarity_matrix[i, j] >= similarity_threshold:
                        # Find the current cluster assignments
                        cluster_i = cluster_assignments[i]
                        cluster_j = cluster_assignments[j]
                        
                        # If they're already in the same cluster, continue
                        if cluster_i == cluster_j:
                            continue
                        
                        # Otherwise merge: assign all patterns in cluster_j to cluster_i
                        for k in range(n_patterns):
                            if cluster_assignments[k] == cluster_j:
                                cluster_assignments[k] = cluster_i
            
            # Renumber clusters to be consecutive integers starting from 0
            unique_clusters = sorted(set(cluster_assignments))
            cluster_map = {old: new for new, old in enumerate(unique_clusters)}
            cluster_assignments = [cluster_map[c] for c in cluster_assignments]
            
            # Group patterns by cluster
            pattern_clusters = {}
            for i, cluster_id in enumerate(cluster_assignments):
                if cluster_id not in pattern_clusters:
                    pattern_clusters[cluster_id] = []
                pattern_clusters[cluster_id].append(pattern_list[i])

            print(f"Patterns clustered into {len(pattern_clusters)} groups using similarity threshold {similarity_threshold}.")

        self.write_to_log(pattern_clusters, 'pattern_clusters')
        return pattern_clusters

    async def generate_regex_patterns(self, patterns: list[str]) -> list[str]:
        o3_mini = ChatOpenAI(model="o3-mini", reasoning_effort = 'low', seed = 42)
        json_llm = o3_mini.bind(response_format={'type': 'json_object'})
        
        # Create a prompt to generate regex patterns for abbreviation extraction
        prompt = f"""
        You are tasked with generating an optimized regex pattern from similar input patterns. Your goal is to create a regex that can match abbreviations and their definitions in technical documents while maintaining specific named groups.

    Here are the input regex patterns:

    <regex_patterns>
    {patterns}
    </regex_patterns>

    Follow these steps to complete the task:

    1. Carefully analyze the provided patterns, identifying common elements and structures.
    2. Note any differences between the patterns, especially optional or variable parts.
    3. Create a new regex pattern that encompasses all the variations found in the input patterns.
    4. Ensure that your new pattern maintains the <ABBR> and <FULL_NAME> named groups, as they are crucial for extracting specific context from longer content.
    5. Optimize the pattern to effectively match abbreviations and their definitions in technical documents. Consider common formats and variations that might appear in such documents.
    6. If possible, simplify the pattern without losing its effectiveness or the required named groups.

    Keep these points in mind when creating the final regex pattern(s):
    - The pattern should be flexible enough to match various forms of abbreviations and their definitions.
    - It should handle optional words or variations in phrasing (e.g., "stands for" vs. "might stands for").
    - The pattern should work well with technical language and formatting.

    After generating the optimized regex pattern(s), format your response as a JSON object with a 'patterns' key containing an array of the most effective regex patterns. Each pattern in the array should be a string representing a complete regex.

    Your final output should only include the JSON object with the patterns, formatted like this:

    <answer>
    {{
    "patterns": [
        "regex_pattern_1",
        "regex_pattern_2"
    ]
    }}
    </answer>

    Do not include any explanations or additional text outside of the JSON object in your final answer. Ensure that your regex patterns are properly escaped within the JSON string format.
        """
        
        ai_msg = json_llm.invoke(prompt)
        print(ai_msg)
        try:
            # Parse the JSON response from the content field
            response_dict = json.loads(ai_msg.content)
            regex_patterns = response_dict.get('patterns', [])
            prod_patterns = [pattern.replace('\\b', '') for pattern in regex_patterns]
            return prod_patterns
        except json.JSONDecodeError:
            print("Error parsing JSON response")
            return []
        except Exception as e:
            print(f"Unexpected error: {e}")
            return []

    def build_abbr_dict(self, processed_graph: dict) -> dict[str, list[str]]:
        regex_list = self.load_log()['regex_for_pattern_clusters']['data']
        dicts = defaultdict(list)
        for entity_name, entity_data in processed_graph.items():
            description = entity_data['description']
            des_arr = description.split('\n')
            first_line = entity_name + ": " + des_arr[0]
            des_arr = [first_line] + des_arr
            for des in des_arr:
                for pattern in regex_list:
                    match = re.search(pattern, des)
                    if match:
                        # Use named groups from the regex pattern
                        abbr = match.group('ABBR')
                        full_name = match.group('FULL_NAME')
                        if len(abbr) > len(full_name):
                            abbr, full_name = full_name, abbr
                        # full_name_uppers = ''.join(c for c in full_name if c.isupper())
                        if full_name not in dicts[abbr]:
                            dicts[abbr].append(full_name)

        llm_abbr_dict = self.load_log()['llm_abbr_dict']['data']
        for abbr, names in llm_abbr_dict.items():
            if not names:
                continue
            if abbr not in dicts:
                dicts[abbr] = names
            else:
                dicts[abbr].extend(names)
        
        # Merge abbreviations that are the same when uppercased
        merged_dicts = {}
        for abbr, expansions in dicts.items():
            uppercase_abbr = abbr.upper()
            if uppercase_abbr in merged_dicts:
                for expansion in expansions:
                    if expansion not in merged_dicts[uppercase_abbr]:
                        merged_dicts[uppercase_abbr].append(expansion)
            else:
                merged_dicts[uppercase_abbr] = expansions.copy()

        # Replace the original dictionary with the merged one
        abbr_dicts = merged_dicts
        deleted_keys = []
        for abbr, names in abbr_dicts.items():
            abbr_dicts[abbr] = [name for name in names if self.normalize_text(name) != self.normalize_text(abbr)]
            abbr_dicts[abbr] = list(set(abbr_dicts[abbr]))
            if not abbr_dicts[abbr]:
                deleted_keys.append(abbr)
        for key in deleted_keys:
            del abbr_dicts[key]
        self.write_to_log(abbr_dicts, 'abbr_dict')
        return abbr_dicts

    async def process_entity_graph(self, input_graph_file='./entity_graph.json', output_graph_file='./processed_entity_graph.json'):
        """
        Process the entity graph with the new alias mapping approach.
        """
        entity_graph = self.load_entity_graph(input_graph_file)
        print(f"Before processing: the number of nodes is {len(entity_graph)}")
        processed_graph = self.merge_duplicate_nodes(entity_graph)

        # Extract abbreviations from descriptions
        entity_names = list(processed_graph.keys())
        log = self.load_log()
        if "identified_abbreviations" in log:
            identified_abbreviations = log["identified_abbreviations"]["data"]
        else:
            batch_size = 40
            tasks = []
            for i in range(0, len(entity_names), batch_size):
                batch = entity_names[i:i+batch_size]
                tasks.append(self.extract_abbreviations(batch))

            identified_abbreviations = []
            for result in await asyncio.gather(*tasks):
                identified_abbreviations.extend(result)
            
            identified_abbreviations = list(set(identified_abbreviations))
            print("Identified abbreviations:")
            pprint.pprint(identified_abbreviations)
            self.write_to_log(identified_abbreviations, 'identified_abbreviations')
            print("Identified abbreviations written to log.json")

        # Build llm abbreviation dictionary and pattern list
        batch_size = 50
        tasks = []
        for i in range(0, len(identified_abbreviations), batch_size):
            batch = list(identified_abbreviations)[i:i+batch_size]
            abbrev_to_descriptions = {abbr: processed_graph[abbr]['description'] for abbr in batch}
            tasks.append(self.extract_pattern(abbrev_to_descriptions))
        pattern_list = []
        llm_abbr_dict = {}
        for result in await asyncio.gather(*tasks):
            print("This is the result:", result) 
            pattern_list.extend(result['regex_patterns'])
            llm_abbr_dict.update(result['abbreviation_expansions'])
        pattern_list = [pattern.replace('\\b', '') for pattern in pattern_list]
        self.write_to_log(llm_abbr_dict, 'llm_abbr_dict')
        self.write_to_log(pattern_list, 'pattern_list')
        print("LLM abbreviation dictionary and pattern list written to log.json")

        # Cluster patterns by similarity and generate optimized regex patterns
        pattern_clusters = self.cluster_patterns_by_similarity(pattern_list=list(set(pattern_list)))
        regex_for_pattern_clusters = []
        tasks = []
        for pattern in pattern_clusters.values():
            pattern = list(set(pattern))
            if len(pattern) > 1:
                tasks.append(self.generate_regex_patterns(pattern))
            else:
                regex_for_pattern_clusters.append(pattern[0])
        for result in await asyncio.gather(*tasks):
            for regex_pattern in result:
                regex_for_pattern_clusters.append(regex_pattern)
        self.write_to_log(regex_for_pattern_clusters, 'regex_for_pattern_clusters')
        print("Regex for pattern clusters written to log.json")

        # Build abbreviation dictionary then use it to merge abbreviation nodes to 
        abbr_dict = self.build_abbr_dict(processed_graph=processed_graph)
        self.merge_abbreviation_nodes(processed_graph, abbr_dict)
        self.remove_self_loops(processed_graph)
        print(f"After processing: the number of nodes is {len(processed_graph)}")
        self.save_graph(output_graph_file, processed_graph)

if __name__ == "__main__":
    processor = EntityGraphProcessor()
    # sensitive_keys = ['OPENAI_API_KEY', 'SECRET_KEY', 'DATABASE_PASSWORD']
    # processor.display_env_variables(exclude_keys=sensitive_keys)
    # processor.clear_log()
    start_time = time.time()
    nest_asyncio.apply()
    asyncio.run(processor.process_entity_graph())
    end_time = time.time()
    print(f"Total time taken: {end_time - start_time} seconds")
