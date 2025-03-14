import json
from typing import Dict
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph
from agentic_flow_construction import FlowConstructor

class QueryProcessor:
    def __init__(self, subgraph_distance: int):
        self.json_file_path = "./retrieved_chunks_15.json"
        self.flow_constructor = FlowConstructor()
        self.flow_constructor.set_subgraph_distance(subgraph_distance)
        self.agentic_flow = self.flow_constructor.create_agentic_flow(subgraph_distance=subgraph_distance)
        self.renewd_question = ""

    def set_renewd_question(self):
        replaced_term = self.flow_constructor.flow_operations.replaced_term
        for key, value in replaced_term.items():
            self.renewd_question = self.renewd_question.replace(key, value)
    
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
            with open(self.json_file_path, 'r') as f:
                data = json.load(f)
            
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
                print(f"Output from node '{key}':")
                print("---")
                print(value)
                result.append(value)    
            print("\n---\n")
        return result

    async def ask_question(self, question: str) -> Dict:
        """
        Ask a question and get entity-based analysis.
        
        Args:
            question (str): The question to analyze
        
        Returns:
            Dict: Analysis results including entities and their information
        """
        chunk = self.find_chunk_for_question(question)
        self.renewd_question = question
        # chunk = "Hello"
        # if not chunk:
        #     return {"messages": [AIMessage(content="No relevant chunk found for this question")]}
        res = await self.process_question_and_chunks(question, chunk)
        self.set_renewd_question()
        return res, self.renewd_question
