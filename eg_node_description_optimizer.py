import json
import numpy as np
import asyncio
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Any, Tuple
from dotenv import load_dotenv, find_dotenv
from openai import OpenAI, AsyncOpenAI
from langchain_openai import ChatOpenAI
from langchain.callbacks.tracers.langchain import wait_for_all_tracers
from tqdm import tqdm
import time

class NodeDescriptionOptimizer:
    def __init__(self):
        """
        Initialize the NodeDescriptionOptimizer.
        """
        env_path = find_dotenv()
        print(f".env file found at: {env_path}")
        load_dotenv()
        print("Environment variables loaded.")
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
        self.client = OpenAI()
        try:
            self.llm.invoke("Hello, World!")
        finally:
            wait_for_all_tracers()
    
    def load_entity_graph(self, file_path: str) -> Dict[str, Any]:
        """Load entity graph from a JSON file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading entity graph: {e}")
            return {}
    
    def save_entity_graph(self, entity_graph: Dict[str, Any], file_path: str) -> None:
        """Save the optimized entity graph to a JSON file."""
        with open(file_path, 'w') as f:
            json.dump(entity_graph, f, indent=2)
    
    def split_into_sentences(self, text: str) -> List[str]:
        """Split a text into sentences."""
        sentences = text.split("\n")
        return [s.strip() for s in sentences if s.strip()]
    
    def get_embeddings(self, sentences: List[str]) -> np.ndarray:
        """Get embeddings for a list of sentences asynchronously."""
        client = OpenAI()
        response = client.embeddings.create(
            input=sentences,
            model="text-embedding-3-small"
        )
        return np.array([data.embedding for data in response.data])
    
    async def async_fuse_sentences_with_llm(self, sentences: List[str]) -> str:
        """Use LLM to fuse similar sentences asynchronously."""
        if not sentences:
            return ""
        
        if len(sentences) == 1:
            return sentences[0]
        if sentences[0] == "<|COMPLETE|>":
            return ""
        
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
        prompt = f"""
        The following sentences contain similar information:
        
        {' '.join([f'- {s}' for s in sentences])}
        
        Please create a single concise sentence that captures the key information from these similar sentences.
        Avoid redundancy and focus on clarity and brevity.
        """
        
        try:
            response = llm.invoke(prompt)
            return response.content
        except Exception as e:
            print(f"Error using LLM for fusion: {e}")
            return sentences[0]
    
    def cluster_sentences_by_similarity(self, sentences: List[str], similarity_threshold: float = 0.9) -> List[List[str]]:
        """Cluster sentences based on cosine similarity of their embeddings asynchronously."""
        if len(sentences) <= 1:
            return [sentences]
        
        embeddings = self.get_embeddings(sentences)
        similarity_matrix = cosine_similarity(embeddings)
        
        # Initialize clusters
        clusters = []
        used_indices = set()
        
        # For each sentence
        for i in range(len(sentences)):
            if i in used_indices:
                continue
                
            # Find all sentences similar to this one
            cluster = [i]
            used_indices.add(i)
            
            for j in range(i + 1, len(sentences)):
                if j not in used_indices and similarity_matrix[i, j] >= similarity_threshold:
                    cluster.append(j)
                    used_indices.add(j)
            
            # Add the actual sentences to the cluster
            clusters.append([sentences[idx] for idx in cluster])
        
        return clusters
    
    async def async_optimize_node_description(self, description: str, similarity_threshold: float = 0.9) -> str:
        """Optimize a single node description asynchronously."""
        # Split into sentences
        sentences = self.split_into_sentences(description)
        
        if len(sentences) <= 2:
            return description  # No need to optimize very short descriptions
        
        # Cluster similar sentences using similarity threshold
        sentence_clusters = self.cluster_sentences_by_similarity(sentences, similarity_threshold)
        
        # Fuse sentences in each cluster
        tasks = []
        fused_sentences = []
        for cluster in sentence_clusters:
            task = self.async_fuse_sentences_with_llm(cluster)
            tasks.append(task)
        
        fused_sentences = await asyncio.gather(*tasks)
        optimized_description = "\n".join(fused_sentences)
        return optimized_description
    
    async def async_optimize_entity_graph(self, entity_graph: Dict[str, Any], similarity_threshold: float = 0.9, batch_size: int = 10) -> Dict[str, Any]:
        """Optimize descriptions for all nodes in the entity graph asynchronously."""
        # Create a deep copy to avoid modifying the original
        optimized_graph = json.loads(json.dumps(entity_graph))
        
        # Get all nodes that have descriptions
        nodes_to_process = []
        for node_name, node_data in optimized_graph.items():
            if "description" in node_data and node_data["description"]:
                nodes_to_process.append((node_name, node_data))
        
        print(f"Processing {len(nodes_to_process)} nodes with descriptions")
        
        # Process nodes in batches to avoid overwhelming the API
        for i in tqdm(range(0, len(nodes_to_process), batch_size)):
            batch = nodes_to_process[i:i+batch_size]
            
            # Create tasks for this batch
            tasks = []
            for node_name, node_data in batch:
                task = self.async_optimize_node_description(node_data["description"], similarity_threshold)
                tasks.append((node_name, task))
            
            # Wait for all tasks in this batch to complete
            for node_name, task in tasks:
                optimized_description = await task
                optimized_graph[node_name]["description"] = optimized_description
        
        return optimized_graph

    def optimize_entity_graph(self, entity_graph: Dict[str, Any], similarity_threshold: float = 0.9) -> Dict[str, Any]:
        """
        Optimize descriptions for all nodes in the entity graph.
        This is a synchronous wrapper around the async implementation.
        """
        return asyncio.run(self.async_optimize_entity_graph(entity_graph, similarity_threshold))

    def fuse_sentences_with_llm(self, sentences: List[str]) -> str:
        """Use LLM to fuse similar sentences asynchronously."""
        if not sentences:
            return ""
        
        if len(sentences) == 1:
            return sentences[0]
        
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
        prompt = f"""
        The following sentences contain similar information:
        
        {' '.join([f'- {s}' for s in sentences])}
        
        Please create a single concise sentence that captures the key information from these similar sentences.
        Avoid redundancy and focus on clarity and brevity.
        """
        
        try:
            response = llm.invoke(prompt)
            return response.content
        except Exception as e:
            print(f"Error using LLM for fusion: {e}")
            return sentences[0]

    def optimize_single_node(self, entity_graph: Dict[str, Any], node_name: str, similarity_threshold: float = 0.9) -> Dict[str, Any]:
        """
        Optimize the description of a single node by name.
        
        Args:
            entity_graph: The original entity graph
            node_name: The name of the node to optimize
            similarity_threshold: Threshold for sentence similarity clustering
            
        Returns:
            Dictionary with original and optimized node data
        """
        # Create a deep copy to avoid modifying the original
        graph_copy = json.loads(json.dumps(entity_graph))
        
        if node_name not in graph_copy.keys():
            print(f"Node '{node_name}' not found in the entity graph.")
            return {"error": f"Node '{node_name}' not found"}
        
        node_data = graph_copy[node_name]
        
        if "description" not in node_data or not node_data["description"]:
            print(f"Node '{node_name}' does not have a description.")
            return {"error": f"Node '{node_name}' has no description"}
        
        original_description = node_data["description"]
        
        # Use the synchronous version for a single node test
        sentences = self.split_into_sentences(original_description)
        
        print(f"Node '{node_name}' has {len(sentences)} sentences.")
        print("Original sentences:")
        for i, sentence in enumerate(sentences):
            print(f"{i+1}. {sentence}")
        
        # Get embeddings and cluster
        response = self.client.embeddings.create(
            input=sentences,
            model="text-embedding-3-small"
        )
        embeddings = np.array([data.embedding for data in response.data])
        similarity_matrix = cosine_similarity(embeddings)
        
        # Print similarity matrix for inspection
        print("\nSimilarity Matrix:")
        np.set_printoptions(precision=2, suppress=True)
        print(similarity_matrix)
        
        # Cluster sentences
        clusters = []
        used_indices = set()
        
        for i in range(len(sentences)):
            if i in used_indices:
                continue
                
            cluster = [i]
            used_indices.add(i)
            
            for j in range(i + 1, len(sentences)):
                if j not in used_indices and similarity_matrix[i, j] >= similarity_threshold:
                    cluster.append(j)
                    used_indices.add(j)
            
            clusters.append([sentences[idx] for idx in cluster])
        
        print(f"\nFound {len(clusters)} clusters with similarity threshold {similarity_threshold}:")
        for i, cluster in enumerate(clusters):
            print(f"Cluster {i+1} ({len(cluster)} sentences):")
            for sentence in cluster:
                print(f"  - {sentence}")
        
        # Fuse sentences in each cluster
        fused_sentences = []
        for i, cluster in enumerate(clusters):
            if len(cluster) == 1:
                fused = cluster[0]
                print(f"\nCluster {i+1} has only one sentence, keeping as is.")
            else:
                print(f"\nFusing cluster {i+1}...")
                fused = self.fuse_sentences_with_llm(cluster)
                print(f"Original sentences:")
                for s in cluster:
                    print(f"  - {s}")
                print(f"Fused result: {fused}")
            
            fused_sentences.append(fused)
        
        # Combine fused sentences
        optimized_description = "\n".join(fused_sentences)
        
        # Print statistics
        original_chars = len(original_description)
        optimized_chars = len(optimized_description)
        reduction = original_chars - optimized_chars
        reduction_percent = (reduction / original_chars) * 100 if original_chars > 0 else 0
        
        print("\nOptimization Results:")
        print(f"Original description ({original_chars} chars):")
        print(original_description)
        print(f"\nOptimized description ({optimized_chars} chars):")
        print(optimized_description)
        print(f"\nReduction: {reduction} chars ({reduction_percent:.2f}%)")
        
        return {
            "node_name": node_name,
            "original_description": original_description,
            "optimized_description": optimized_description,
            "original_chars": original_chars,
            "optimized_chars": optimized_chars,
            "reduction": reduction,
            "reduction_percent": reduction_percent
        }

    def compare_qa_on_descriptions(self, original_description: str, optimized_description: str, question: str) -> Dict[str, Any]:
        """
        Compare question answering on original vs optimized descriptions.
        
        Args:
            original_description: The original node description
            optimized_description: The optimized node description
            question: The question to answer
            
        Returns:
            Dictionary with both answers and analysis
        """
        print(f"\nComparing QA for question: '{question}'")
        
        # Create prompts for both descriptions
        original_prompt = f"""
        Based only on the following information, please answer the question.
        
        Information:
        {original_description}
        
        Question: {question}
        """
        
        optimized_prompt = f"""
        Based only on the following information, please answer the question.
        
        Information:
        {optimized_description}
        
        Question: {question}
        """
        
        # Get answers using LLM
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)  # Using a more capable model for QA
        
        print("Getting answer from original description...")
        original_answer = llm.invoke(original_prompt).content
        
        print("Getting answer from optimized description...")
        optimized_answer = llm.invoke(optimized_prompt).content
        
        # Compare the answers
        comparison_prompt = f"""
        I have two answers to the question: "{question}"
        
        Answer 1 (based on original text):
        {original_answer}
        
        Answer 2 (based on optimized text):
        {optimized_answer}
        
        Please analyze these answers and tell me:
        1. Are they substantively the same or different?
        2. What key information is present in both answers?
        3. What information, if any, is missing from Answer 2 that was present in Answer 1?
        4. What information, if any, is present in Answer 2 but not in Answer 1?
        5. Overall, does Answer 2 preserve the essential information needed to answer the question?
        
        Provide a detailed analysis.
        """
        
        print("Analyzing answers...")
        judge = ChatOpenAI(model="o3-mini", reasoning_effort='low', seed=42)
        analysis = judge.invoke(comparison_prompt).content
        
        result = {
            "question": question,
            "original_answer": original_answer,
            "optimized_answer": optimized_answer,
            "analysis": analysis
        }
        
        # Print results
        print("\nQuestion:", question)
        print("\nAnswer from original description:")
        print(original_answer)
        print("\nAnswer from optimized description:")
        print(optimized_answer)
        print("\nAnalysis:")
        print(analysis)
        
        return result


# Example usage
if __name__ == "__main__":
    start_time = time.time()
    optimizer = NodeDescriptionOptimizer()
    
    # Load entity graph
    entity_graph = optimizer.load_entity_graph("./processed_entity_graph.json")
    print("Starting optimization on a single node")

#=============Single node optimization=============
    # # Test optimization on a single node
    # node_name = "HOMEPLUG GREEN PHY"
    # result = optimizer.optimize_single_node(entity_graph, node_name)
    # print("Optimization complete")

    # # If you want to save the result to a file
    # with open(f"./optimized_node_{node_name.replace(' ', '_')}.json", "w") as f:
    #     json.dump(result, f, indent=2)

    # # Test QA on the original vs optimized descriptions
    # qa_result = optimizer.compare_qa_on_descriptions(
    #     result["original_description"],
    #     result["optimized_description"],
    #     "What is HPGP?"
    # )
    
    # # Save QA results
    # with open(f"./qa_comparison_{node_name.replace(' ', '_')}.json", "w") as f:
    #     json.dump(qa_result, f, indent=2)



#=============Whole graph optimization=============
    # Optimize node descriptions
    # optimized_graph = optimizer.optimize_entity_graph(entity_graph)
    
    # # Save optimized graph
    # optimizer.save_entity_graph(optimized_graph, "./optimized_entity_graph.json")
    optimized_graph = optimizer.load_entity_graph("./optimized_entity_graph.json")
    end_time = time.time()
    print(f"Optimization completed in {end_time - start_time:.2f} seconds")
    # Print statistics
    original_chars = sum(len(node.get("description", "")) for node_name, node in entity_graph.items())
    optimized_chars = sum(len(node.get("description", "")) for node_name, node in optimized_graph.items())
    print(original_chars)
    print(optimized_chars)
    print(f"Original total characters: {original_chars}")
    print(f"Optimized total characters: {optimized_chars}")
    print(f"Reduction: {original_chars - optimized_chars} characters ({(1 - optimized_chars/original_chars)*100:.2f}%)")