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
        if "subgraph_distance" in kwargs:
            self.flow_constructor.set_subgraph_distance(kwargs["subgraph_distance"])
        if "graph_path" in kwargs:
            self.flow_constructor.flow_operations.set_graph_path(kwargs["graph_path"])
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

    def combine_entity_descriptions(self, data):
        """
        Combines the 'relationship', 'description', and 'chunk_context' fields
        for each entity in the JSON file into a single detailed description.

        Args:
            input_file_path (str): The path to the entity_chunks.json file.

        Returns:
            dict: A dictionary with entity names as keys and their combined descriptions as values.
        """
    
        entities_chunks = {}
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
            for entity, details in entities.items():
                
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

    def generate_final_prompt(self, data, question: str):
        combined_dict = self.combine_entity_descriptions(data)
        final_prompt = ""
        for query_entity, entity_chunks in combined_dict.items():
            final_prompt += f"================================= Entity Chunks for {query_entity} =================================\n"
            for entity, description in entity_chunks.items():
                final_prompt += f"Entity: {entity}\nDescription: {description}\n{'-'*80}\n"

        final_prompt += f"You need to answer the following question as more details as possible based on the provided information above\n Question: {question}"
        
        return final_prompt

    def generate_answer(self, data, question: str):
        final_prompt = self.generate_final_prompt(data, question)
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
        
        # Check if the content is an error message
        if content.startswith("No matching entities") or content.startswith("No entities were identified") or content.startswith("No valid entities"):
            return f"I couldn't find information about the entities in your question. {content}"
        
        # Try to parse the content as JSON
        try:
            entity_chunks = json.loads(content)
            self.write_chunk_to_file(question, entity_chunks)
            self.set_renewd_question()
            print("Renewed question: ", self.renewd_question)
            answer = self.generate_answer(entity_chunks, self.renewd_question)
            return answer
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON: {e}")
            return f"There was an error processing your question: {str(e)}"
