import json
from typing import Dict
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph
from agentic_flow_construction import FlowConstructor
from langchain_openai import ChatOpenAI
import re

class QueryProcessor:
    def __init__(self, **kwargs):
        self.retrieved_chunks_path = "./retrieved_chunks_15.json"
        self.entities_chunks_path = "./entities_chunks.json"
        self.flow_constructor = FlowConstructor()
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        # self.llm = ChatOpenAI(model="o4-mini", reasoning_effort="medium")
        if "subgraph_distance" in kwargs:
            self.flow_constructor.set_subgraph_distance(kwargs["subgraph_distance"])
        if "graph_path" in kwargs:
            self.flow_constructor.flow_operations.set_graph_path(kwargs["graph_path"])
            self.entity_graph_path = kwargs["graph_path"]
        self.agentic_flow = self.flow_constructor.create_agentic_flow()
        self.renewd_question = ""

    def set_renewd_question(self):
        replaced_term = self.flow_constructor.flow_operations.replaced_term
        for key, value in replaced_term.items():
            self.renewd_question = self.renewd_question.replace(key, value)
    
    def set_graph_path(self, path: str):
        self.flow_constructor.flow_operations.set_graph_path(path)

    def load_json(self, path: str):
        with open(path, 'r') as f:
            return json.load(f)

    def find_chunk_for_question(self, question: str) -> str:
        """
        Find the corresponding chunk for a given question from the JSON file.
    
        Args:
            question (str): The question to look for
            json_file_path (str): Path to the JSON file containing chunks
        
        Returns:
            str: The corresponding chunk text, or empty string if not found
        """
        try:
            data = self.load_json(self.retrieved_chunks_path)
            
            # First try exact match
            for entry in data:
                if entry["question"].lower().strip() == question.lower().strip():
                    return entry.get("prompt", "")
            
            # If no exact match, try fuzzy matching
            from difflib import SequenceMatcher
            
            def similarity(a, b):
                return SequenceMatcher(None, a.lower(), b.lower()).ratio()
            
            best_match = None
            best_score = 0
            
            for entry in data:
                score = similarity(entry["question"], question)
                if score > best_score and score > 0.8:  # 0.8 threshold for similarity
                    best_score = score
                    best_match = entry
            
            if best_match:
                return best_match.get("prompt", "")
            
            return ""
        
        except Exception as e:
            print(f"Error reading JSON file: {e}")
            return ""
        
    async def process_question_and_chunks(self, question: str, chunks: str) -> Dict:
        """
        Process a question and its corresponding chunks through the agent.
        
        Args:
            question (str): The question to analyze
            chunks (str): The corresponding chunk text containing background information
        
        Returns:
            Dict: Processed information about relevant entities
        """
        inputs = {
            "messages": [HumanMessage(content=f"Question: {question}\n\nBackground Chunks:\n{chunks}")]
        }
        result = []
        async for output in self.agentic_flow.astream(inputs, stream_mode="updates"):
        # stream_mode="updates" yields dictionaries with output keyed by node name
            for key, value in output.items():
                # print(f"Output from node '{key}':")
                # print("---")
                # print(value)
                result.append(value)    
            # print("\n---\n")
        return result

    def extract_intention(self, question: str):
        prompt = f"""
Your task is to recognize the user's intention in a given query. This is crucial for understanding the type of information the user is seeking and providing an appropriate response. There are three main categories of user intentions that you need to identify:

1. General Information Query: This category includes most common questions that don't fall into the other two categories. These are typically questions seeking basic information about a topic.

2. Comparison Query: This category includes questions that ask about differences or comparisons between two or more subjects.

3. Commonality Query: This category includes questions that ask about shared characteristics or similarities between two or more subjects.

To analyze the user query, follow these steps:

1. Carefully read the entire query.
2. Look for key phrases or structures that indicate the query type.
3. Consider the overall context and what the user is trying to learn.

After analyzing the query, return a JSON object with the following structure:
- Key: "category" - Value: The identified intention category name
- Key: "explanation" - Value: A brief explanation of why you classified it this way

Here are examples of each category response:

1. General Information Query:
User Query: "What is artificial intelligence?"
```json
{{
  "category": "General Information Query",
  "explanation": "This query is asking for basic information about artificial intelligence without comparing it to anything else or asking about commonalities."
}}
```

2. Comparison Query:
User Query: "What is the difference between machine learning and deep learning?"
```json
{{
  "category": "Comparison Query",
  "explanation": "This query explicitly asks for the difference between two subjects (machine learning and deep learning), indicating a comparison."
}}
```

3. Commonality Query:
User Query: "What do electric cars and hybrids have in common?"
```json
{{
  "category": "Commonality Query",
  "explanation": "This query directly asks about shared characteristics between two subjects (electric cars and hybrids)."
}}
```

Now, analyze the following user query and provide your response:

<user_query>
{question}
</user_query>

Remember to return a JSON object with keys "category" and "explanation" as shown in the examples.
"""
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0).with_structured_output(method="json_mode")
        response = llm.invoke(prompt)
        category = response.get("category", "")
        explanation = response.get("explanation", "")
        return category, explanation
        


    def get_common_entities_dict(self, data, max_distance=-1):
        """
        Returns a new dictionary containing only the common entities across all chunks,
        maintaining the original structure for each entity
        Args:
            data: dictionary containing entity_chunk_X keys with entity information
        Returns:
            dict: new dictionary with only common entities
        """
        # First find common entity names across all chunks
        chunk_keys = list(data.keys())
        if not chunk_keys:
            return {}
        
        # Get common entity names (like 'EIM', 'AC', etc.)
        common_entities = set(data[chunk_keys[0]].keys())
        for chunk_key in chunk_keys[1:]:
            common_entities = common_entities.intersection(set(data[chunk_key].keys()))
        
        # Create new dictionary with only common entities
        common_dict = {}
        
        # First handle main entities (distance=0)
        for chunk_key in chunk_keys:
            main_entity = next((entity for entity, info in data[chunk_key].items() 
                            if info.get('distance') == 0), None)
            
            if main_entity and (max_distance < 0 or 0 <= max_distance):
                if main_entity not in common_dict:
                    common_dict[main_entity] = data[chunk_key][main_entity]
        
        # Then handle other common entities
        for entity in common_entities:
            if entity in common_dict:  # Skip if already added as main entity
                continue
                
            # Find smallest distance for this entity across all chunks
            smallest_distance = float('inf')
            best_data = None
            
            for chunk_key in chunk_keys:
                entity_data = data[chunk_key].get(entity, {})
                distance = entity_data.get('distance', float('inf'))
                if distance < smallest_distance:
                    smallest_distance = distance
                    best_data = entity_data
            
            # Only include entity if its smallest distance is within limit
            if max_distance < 0 or smallest_distance <= max_distance:
                common_dict[entity] = best_data
        common_dict = {"common_entities": common_dict}
        return common_dict

    def get_unique_entities_dict(self, data, max_distance=-1):
        """
        Returns a new dictionary containing only the unique entities for each chunk,
        maintaining the original structure for each entity
        Args:
            data: dictionary containing entity_chunk_X keys with entity information
        Returns:
            dict: new dictionary with only unique entities per chunk
        """
        chunk_keys = list(data.keys())
        unique_dict = {}
        # For each chunk, find entities that don't appear in any other chunk
        for current_chunk in chunk_keys:
            current_entities = set(data[current_chunk].keys())
            
            # Get entities from all other chunks
            other_entities = set()
            for other_chunk in chunk_keys:
                if other_chunk != current_chunk:
                    other_entities.update(data[other_chunk].keys())
            
            # Get unique entities for this chunk
            unique_entities = current_entities - other_entities
            
            main_entity = next((entity for entity, info in data[current_chunk].items() 
                            if info.get('distance') == 0), None)
            
            # Create new dictionary with unique entities and main entity for this chunk
            unique_dict[current_chunk] = {}
            
            if main_entity:
                unique_dict[current_chunk][main_entity] = data[current_chunk][main_entity]
                
            # Add other unique entities
            for entity in unique_entities:
                if entity != main_entity:  # Don't add main entity twice
                    entity_data = data[current_chunk][entity]
                    distance = entity_data.get('distance', float('inf'))
                    if max_distance < 0 or distance <= max_distance:
                        unique_dict[current_chunk][entity] = entity_data
        
        return unique_dict

    def combine_entity_descriptions(self, data, max_distance=3):
        """
        Combines the 'relationship', 'description', and 'chunk_context' fields
        for each entity in the JSON file into a single detailed description.

        Args:
            data (dict): Dictionary containing entity information
            max_distance (int): Maximum distance to include entities
            
        Returns:
            dict: A dictionary with entity names as keys and their combined descriptions as values.
        """
        entities_chunks = {}
        # Process entities up to max_distance
        for entity_chunk, entities in data.items():
            first_one = True
            main_entity = ""
            for entity, details in entities.items():
                if details.get('distance') == 0:
                    if not first_one:
                        main_entity += " & "
                    main_entity += entity
                    first_one = False
                else: break
            entities_chunks[main_entity] = {}
            
            # Process entities up to max_distance
            for entity, details in entities.items():
                if details.get('distance') > max_distance:
                    continue
                
                combined_parts = []
                
                # Extract and append 'relationship' if it's not empty
                relationship = details.get('relationship', '').strip()
                if relationship:
                    combined_parts.append(relationship.strip('"'))  # Remove surrounding quotes if present

                # Extract and append 'description' if it's not empty
                description = details.get('description', '').strip()
                if description:
                    # Replace <SEP> with a space or another separator if desired
                    description = description.replace('<SEP>', ' ')
                    combined_parts.append(description.strip('"'))  # Remove surrounding quotes if present

                # Extract and append 'chunk_context' if it's not empty
                chunk_context = details.get('chunk_context', '').strip()
                if chunk_context:
                    combined_parts.append(chunk_context)

                # Join all parts with a space
                combined_description = ' '.join(combined_parts)
                entities_chunks[main_entity][entity] = combined_description

        return entities_chunks

    def format_entity_chunks_prompt(self, data, main_entity):
        chunks_prompt = f"================================= Entity Chunks for {main_entity} =================================\n"
        for entity, description in data.items():
            chunks_prompt += f"Entity: {entity}\nDescription: {description}\n{'-'*80}\n"
        return chunks_prompt

    def extend_entities_from_graph(self, data, max_distance=1):
        """
        Extends the entities in the data by finding their neighbors in the entity graph
        up to a specified distance.

        Args:
            data (dict): Dictionary containing entity_chunk_X keys with entity information
            max_distance (int): Maximum distance to extend entities in the graph
            
        Returns:
            dict: The original data with additional entities added from the graph
        """
        try:
            # Load the entity graph
            entity_graph = self.load_json(self.entity_graph_path)
            if not entity_graph:
                print("Warning: Could not load entity graph, returning original data")
                return data
                
            # Create a deep copy to avoid modifying the original data
            import copy
            extended_data = copy.deepcopy(data)
            
            # For each chunk in the data
            for chunk_key, entities in extended_data.items():
                # Find all entities at the boundary (maximum distance in current data)
                current_max_distance = max([details.get('distance', 0) for entity, details in entities.items()])
                boundary_entities = [entity for entity, details in entities.items() 
                                    if details.get('distance') == current_max_distance]
                
                # Track processed entities to avoid duplicates
                processed_entities = set(entities.keys())
                
                # For each entity at the boundary
                for boundary_entity in boundary_entities:
                    self._extend_entity_neighbors(
                        entity_graph, 
                        extended_data[chunk_key], 
                        boundary_entity, 
                        current_max_distance,
                        processed_entities, 
                        max_depth=max_distance
                    )
                    
            return extended_data
            
        except Exception as e:
            print(f"Error extending entities from graph: {e}")
            return data
            
    def _extend_entity_neighbors(self, entity_graph, entities_dict, entity_name, current_distance, 
                                processed_entities, max_depth=1, current_depth=0):
        """
        Recursively extends entity neighbors from the graph.
        
        Args:
            entity_graph (dict): The loaded entity graph
            entities_dict (dict): The dictionary to add new entities to
            entity_name (str): The current entity to find neighbors for
            current_distance (int): The current distance in the original data
            processed_entities (set): Set of already processed entity names
            max_depth (int): Maximum recursion depth
            current_depth (int): Current recursion depth
        """
        # Base case: reached max depth or entity not in graph
        if current_depth >= max_depth or entity_name not in entity_graph:
            return
            
        # Get connections for this entity
        connections = entity_graph.get(entity_name, {}).get("connections", [])
        
        # Add each connection as a new entity if not already processed
        for connection in connections:
            neighbor_name = connection.get("target", "")
            if not neighbor_name or neighbor_name in processed_entities:
                continue
                
            # Mark as processed
            processed_entities.add(neighbor_name)
            
            # Create new entity details
            new_distance = current_distance + 1
            connection_desc = connection.get("description", "")
            
            # Get entity's own description if available
            neighbor_desc = ""
            if neighbor_name in entity_graph:
                neighbor_desc = entity_graph[neighbor_name].get("description", "")
                
            # Create relationship description
            relationship = f"Related to {entity_name}"
            if connection_desc:
                relationship = connection_desc
                
            # Add the new entity
            entities_dict[neighbor_name] = {
                'distance': new_distance,
                'relationship': relationship,
                'description': neighbor_desc,
                'chunk_context': ""  # No context for graph-derived entities
            }
            
            # Recursively extend from this new entity
            self._extend_entity_neighbors(
                entity_graph, 
                entities_dict, 
                neighbor_name, 
                new_distance,
                processed_entities, 
                max_depth, 
                current_depth + 1
            )

    def generate_final_prompt(self, data, question: str, intention_category: str):
        final_prompt = ""
        extracted_dis = self.flow_constructor.subgraph_distance
        print("This is the extraction range: ", extracted_dis)
        print("This is the intention category: ", intention_category)
        combined_dict = self.combine_entity_descriptions(data, max_distance=extracted_dis)
        for query_entity, entity_chunks in combined_dict.items():
            final_prompt += self.format_entity_chunks_prompt(entity_chunks, query_entity)
        # For General Information Query, no extension needed
        # if intention_category == "General Information Query":
        #     # No extension for general queries
        #     combined_dict = self.combine_entity_descriptions(data, max_distance=extracted_dis)
        #     for query_entity, entity_chunks in combined_dict.items():
        #         final_prompt += self.format_entity_chunks_prompt(entity_chunks, query_entity)

        # elif intention_category == "Comparison Query" or intention_category == "Commonality Query":
        #     # Determine which aspect needs extension based on query type
        #     common_extend = 1 if intention_category == "Commonality Query" else 0
        #     diff_extend = 1 if intention_category == "Comparison Query" else 0
            
        #     common_dis = extracted_dis + 1 if intention_category == "Commonality Query" else extracted_dis
        #     dif_dis = extracted_dis + 1 if intention_category == "Comparison Query" else extracted_dis
            
        #     print("These are common & diff distance: ", common_dis, dif_dis)
            
        #     # First extend the entities if needed
        #     extended_data = data
        #     if common_extend > 0 or diff_extend > 0:
        #         extension_distance = max(common_extend, diff_extend)
        #         extended_data = self.extend_entities_from_graph(data, max_distance=extension_distance)
            
        #     # Then extract common/unique entities from the extended data
        #     common_entities_dict = self.get_common_entities_dict(extended_data, common_dis) if common_dis != -1 else {}
        #     common_chunks = self.combine_entity_descriptions(common_entities_dict, max_distance=common_dis)
        #     common_prompts = {}
        #     num_of_entities = 0
            
        #     for entity_chunk, chunks in common_chunks.items():
        #         num_of_entities += len(chunks.keys())
        #         common_prompts[entity_chunk] = self.format_entity_chunks_prompt(chunks, entity_chunk)
                
        #     unique_entities_dict = self.get_unique_entities_dict(extended_data, dif_dis) if dif_dis != -1 else {}
        #     unique_chunks = self.combine_entity_descriptions(unique_entities_dict, max_distance=dif_dis)
        #     unique_prompts = {}
            
        #     for entity_chunk, chunks in unique_chunks.items():
        #         num_of_entities += len(chunks.keys())
        #         unique_prompts[entity_chunk] = self.format_entity_chunks_prompt(chunks, entity_chunk)
                
        #     for main_entity, chunks in unique_prompts.items():
        #         final_prompt += f"Below is the unique entity information for {main_entity}\n"
        #         final_prompt += chunks
        #     for main_entity, chunks in common_prompts.items():
        #         final_prompt += f"Below is the common entity information for {main_entity}\n"
        #         final_prompt += chunks

        final_prompt += f"You need to answer the following question as more details as possible based on the provided information above\n Question: {question}"
        # print(final_prompt)
        # Save the prompt in a dictionary format with the question as the key
        # try:
        #     with open("final_prompt.json", "r") as f:
        #         prompt_dict = json.load(f)
        # except (FileNotFoundError, json.JSONDecodeError):
        #     prompt_dict = {}
        # prompt_dict[question] = final_prompt
        # with open("final_prompt.json", "w") as f:
        #     json.dump(prompt_dict, f, indent=4)
        return final_prompt

    def generate_answer(self, data, question: str):
        """
        Distinguish the user intention and generate the corressponding prompt
        """
        intention_category, intention_explanation = self.extract_intention(question)
        final_prompt = self.generate_final_prompt(data, question, intention_category)
        response = self.llm.invoke(final_prompt)
        final_answer = response.content
        final_answer = re.sub(
                r'(\d+\. \*\*[^:]+\*\*): ', 
                r'\n### \1\n', 
                final_answer
            )
        print("Token Usage:", response.response_metadata['token_usage'])
        return final_answer

    #can be removed
    def check_if_question_exists(self, question: str):
        existing_data = self.load_json(self.entities_chunks_path)
        for entry in existing_data:
            if entry == question:
                return True
        return False
    
    def write_chunk_to_file(self, question: str, entity_chunks: dict):
        """
        Write a question and its associated entity chunks to the JSON file.
        
        Args:
            question (str): The question associated with the chunk
            chunk (dict): The entity chunk data
        """
        try:
            # First, load existing data
            existing_data = self.load_json(self.entities_chunks_path)    
            question_exists = self.check_if_question_exists(question)
            if not question_exists:
                existing_data[question] = entity_chunks
            # Write the updated data back to the file
            with open(self.entities_chunks_path, 'w') as f:
                json.dump(existing_data, f, indent=4)
            print(f"Successfully saved entity chunks for question: '{question}'")
            
        except Exception as e:
            print(f"Error writing chunk to file: {e}")

    async def ask_question(self, question: str) -> Dict:
        """
        Ask a question and get entity-based analysis.
        
        Args:
            question (str): The question to analyze
        
        Returns:
            Dict: Analysis results including entities and their information
        """
        if self.check_if_question_exists(question):
            return None
            
        chunk = self.find_chunk_for_question(question)
        self.renewd_question = question
        res = await self.process_question_and_chunks(question, chunk)
        
        # Check if we have a valid result
        if not res or len(res) == 0:
            return "No results were returned for this question."
        
        last_message = res[-1].get('messages', None)
        if not last_message:
            return "No message content in the response."
        
        content = last_message.content
        flag = False
        # Check if the content is an error message
        if content.startswith("No matching entities") or content.startswith("No entities were identified") or content.startswith("No valid entities"):
            # return f"I couldn't find information about the entities in your question. {content}"
            flag = True
        # Try to parse the content as JSON
        try:
            if flag:
                prompt = f"""{chunk}\n\nYou need to answer the following question as more details as possible based on the provided information above\n Question: {question}"""
                answer = self.llm.invoke(prompt).content
            else:
                entity_chunks = json.loads(content)
                self.write_chunk_to_file(question, entity_chunks)
                self.set_renewd_question()
                print("Renewed question: ", self.renewd_question)
                answer = self.generate_answer(entity_chunks, self.renewd_question)
            return answer
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON: {e}")
            return f"There was an error processing your question: {str(e)}"
