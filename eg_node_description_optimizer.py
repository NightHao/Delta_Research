import json
import numpy as np
import asyncio
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
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
    
    async def get_embeddings_async(self, sentences: List[str]) -> np.ndarray:
        client = AsyncOpenAI()  # Use AsyncOpenAI
        response = await client.embeddings.create(
            input=sentences,
            model="text-embedding-3-small"
        )
        return np.array([data.embedding for data in response.data])
    
    async def async_fuse_sentences_with_llm(self, sentences: List[str]) -> Tuple[str, str]:
        """Use LLM to fuse similar sentences asynchronously."""
        if not sentences:
            return "", ""
        
        if len(sentences) == 1:
            return "", sentences[0]
        
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
        prompt = f"""
<input_sentences>
{' '.join([f'- {s}' for s in sentences])}
</input_sentences>

You are an expert in sentence fusion and information synthesis. Your task is to merge multiple related sentences into a single, concise sentence that captures all important information. The sentences you need to merge are provided above.

Please follow these steps to complete the task:

1. Analyze the input sentences.
2. Identify key information, common themes, and unique elements from each sentence.
3. Determine which information is redundant and can be removed.
4. Create a structure for the merged sentence that includes all important information.
5. Draft a merged sentence that is concise, clear, and avoids redundancy.
6. Ensure the merged sentence is grammatically correct and preserves the original meaning.

Before providing your final merged sentence, wrap your thought process in <sentence_fusion_process> tags. Follow these steps within the tags:

1. List the key information from each input sentence.
2. Highlight common themes and unique information across sentences.
3. Identify redundant information that can be removed.
4. Consider potential connecting words or phrases to link ideas smoothly.
5. Outline the structure for the merged sentence.
6. Draft an initial merged sentence.
7. Evaluate the clarity and coherence of the merged sentence, making adjustments if necessary.

After completing your thought process, create the final merged sentence.

Your output should be a JSON object with two keys:
1. "process": containing the content of your <sentence_fusion_process> tags
2. "merged_sentence": containing your final merged sentence

Example output structure (using generic content):

{{
  "process": "1. Key information:\n   - Sentence 1: [key points]\n   - Sentence 2: [key points]\n   ...\n2. Common themes: [list themes]\n3. Unique information: [list unique elements]\n4. Redundant information: [list redundancies]\n5. Connecting words/phrases: [list potential connectors]\n6. Merged sentence structure: [outline structure]\n7. Initial draft: [draft sentence]\n8. Evaluation and adjustments: [notes on clarity and coherence]",
  "merged_sentence": "Your final merged sentence goes here."
}}

Please proceed with the sentence fusion task and provide your output in the specified JSON format.
"""

        try:
            response = json.loads((await llm.ainvoke(prompt)).content)
            thought = response.get("process", "")
            merged_sentence = response.get("merged_sentence", "")
            return thought, merged_sentence
        except Exception as e:
            print(f"Error using LLM for fusion: {e}")
            return "", sentences[0]
    
    # def cluster_sentences_by_similarity(self, sentences: List[str], similarity_threshold: float = 0.9) -> List[List[str]]:
    #     """Cluster sentences based on cosine similarity of their embeddings asynchronously."""
    #     if len(sentences) <= 1:
    #         return [sentences]
        
    #     embeddings = self.get_embeddings(sentences)
    #     similarity_matrix = cosine_similarity(embeddings)
        
    #     # Initialize clusters
    #     clusters = []
    #     used_indices = set()
        
    #     # For each sentence
    #     for i in range(len(sentences)):
    #         if i in used_indices:
    #             continue
                
    #         # Find all sentences similar to this one
    #         cluster = [i]
    #         used_indices.add(i)
            
    #         for j in range(i + 1, len(sentences)):
    #             if j not in used_indices and similarity_matrix[i, j] >= similarity_threshold:
    #                 cluster.append(j)
    #                 used_indices.add(j)
            
    #         # Add the actual sentences to the cluster
    #         clusters.append([sentences[idx] for idx in cluster])
        
    #     return clusters
    
    async def cluster_and_fuse_sentences(self, sentences: List[str], similarity_threshold: float = 0.9, max_cluster_size: int = 5, iteration: int = 1, max_iterations: int = 3) -> Tuple[List[str], int]:
        """Cluster sentences and fuse them, ensuring no cluster exceeds max_cluster_size."""
        final_iteration_count = iteration
        if len(sentences) <= 1:
            return sentences, final_iteration_count
        
        # Only print the initial state for the first iteration
        # if iteration == 1:
        #     print(f"\n==== STARTING OPTIMIZATION ====")
        #     print(f"Initial sentence count: {len(sentences)}")
        #     print(f"Similarity threshold: {similarity_threshold}")
        #     print(f"Max cluster size: {max_cluster_size}")
        #     print(f"Max iterations: {max_iterations}")
        
        # print(f"\n==== ITERATION {iteration} ====")
        # print(f"Processing {len(sentences)} sentences...")
        
        # Step 1: Create initial clusters
        embeddings = await self.get_embeddings_async(sentences)
        similarity_matrix = cosine_similarity(embeddings)
        
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
        
        # Separate clusters by size for processing
        large_clusters = [c for c in clusters if len(c) > max_cluster_size]
        normal_clusters = [c for c in clusters if len(c) <= max_cluster_size]
        
        # print(f"\nIteration {iteration} initial clusters:")
        # print(f"- Total clusters: {len(clusters)}")
        # print(f"- Single-sentence clusters: {len([c for c in clusters if len(c) == 1])}")
        # print(f"- Multi-sentence clusters: {len([c for c in clusters if len(c) > 1])}")
        # print(f"- Oversized clusters (>{max_cluster_size}): {len(large_clusters)}")
        
        # Step 2: Process clusters that are too large
        final_clusters = normal_clusters.copy()  # Start with normal clusters 
        
        # Process large clusters
        for large_cluster in large_clusters:
            # print(f"Processing large cluster with {len(large_cluster)} sentences...")
            # Split large clusters into chunks of max_cluster_size
            for j in range(0, len(large_cluster), max_cluster_size):
                subcluster = large_cluster[j:j+max_cluster_size]
                final_clusters.append(subcluster)
        
        # print(f"After splitting: {len(final_clusters)} clusters (from {len(clusters)} original clusters)")
        
        # Step 3: Fuse sentences in each cluster
                # Step 3: Fuse sentences in each cluster
        fusion_tasks = []
        clusters_to_fuse = []
        single_sentence_results = []
        
        for cluster in final_clusters:
            if len(cluster) > 1:  # Only fuse multi-sentence clusters
                task = self.async_fuse_sentences_with_llm(cluster)
                fusion_tasks.append(task)
                clusters_to_fuse.append(cluster)
            else:  # This is a single-sentence cluster
                single_sentence_results.append(cluster[0])
        
        # Parallel execution of all fusion tasks
        if fusion_tasks:
            fusion_results = await asyncio.gather(*fusion_tasks)
        else:
            fusion_results = []
        
        # Process results
        results = list(single_sentence_results)  # Start with single-sentence clusters
        fused_count = 0
        
        for i, (thought, merged_sentence) in enumerate(fusion_results):
            original_cluster = clusters_to_fuse[i]
            
            # Print info about the fusion if U want to check the process
            # print(f"Original sentences:")
            # for sentence in original_cluster:
            #     print(f"  - {sentence}")
            # print(f"Fused result: {merged_sentence}")
            # print(f"Thought process:\n  {thought}")
            
            results.append(merged_sentence)
            fused_count += 1
        
        # Step 4: Check if further optimization is needed and iteration limit not reached
        if len(results) < len(sentences) and len(results) > 1 and iteration < max_iterations:
            # print(f"\nIteration {iteration} summary:")
            # print(f"- Sentences: {len(sentences)} → {len(results)} ({len(sentences)-len(results)} merged)")
            # print(f"- Clusters fused: {fused_count}")
            
            # Recursively continue with next iteration
            results, final_iteration_count = await self.cluster_and_fuse_sentences(
                results, 
                similarity_threshold, 
                max_cluster_size, 
                iteration + 1, 
                max_iterations
            )

        # Check if we stopped due to iteration limit
        # if iteration >= max_iterations and len(results) > 1:
        #     print(f"\n⚠️ Reached maximum iterations limit ({max_iterations}). Stopping optimization process.")

        # if iteration == 1:
        #     print(f"\n==== OPTIMIZATION COMPLETE ====")
        #     print(f"Starting sentences: {len(sentences)}")
        #     print(f"Final sentences: {len(results)}")
        #     print(f"Total iterations: {final_iteration_count}")
        #     print(f"Reduction: {len(sentences) - len(results)} sentences ({((len(sentences) - len(results)) / len(sentences) * 100):.2f}%)")

        return results, final_iteration_count
        
    async def async_optimize_node_description(self, description: str, similarity_threshold: float = 0.9, max_cluster_size: int = 5) -> str:
        """Optimize a single node description asynchronously."""
        sentences = self.split_into_sentences(description)
        
        if len(sentences) <= 2:
            return description  # No need to optimize very short descriptions
        
        optimized_sentences, _ = await self.cluster_and_fuse_sentences(sentences, similarity_threshold, max_cluster_size)
        return "\n".join(optimized_sentences)
    
    async def async_optimize_entity_graph(self, entity_graph: Dict[str, Any], similarity_threshold: float = 0.9, max_cluster_size: int = 5, batch_size: int = 10) -> Dict[str, Any]:
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
            
            # Create tasks and track node names
            tasks = []
            node_names = []
            
            for node_name, node_data in batch:
                task = self.async_optimize_node_description(node_data["description"], similarity_threshold, max_cluster_size)
                tasks.append(task)
                node_names.append(node_name)
            
            # Process all tasks in parallel
            batch_results = await asyncio.gather(*tasks)
            
            # Update optimized graph with results (preserving the order)
            for node_name, optimized_description in zip(node_names, batch_results):
                optimized_graph[node_name]["description"] = optimized_description
        
        return optimized_graph

    def optimize_entity_graph(self, entity_graph: Dict[str, Any], similarity_threshold: float = 0.9, max_cluster_size: int = 5) -> Dict[str, Any]:
        """
        Optimize descriptions for all nodes in the entity graph.
        This is a synchronous wrapper around the async implementation.
        """
        return asyncio.run(self.async_optimize_entity_graph(entity_graph, similarity_threshold, max_cluster_size))

    async def optimize_single_node(self, entity_graph: Dict[str, Any], node_name: str, similarity_threshold: float = 0.9, max_cluster_size: int = 5) -> Dict[str, Any]:
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
                
        optimized_sentences, _ = await self.cluster_and_fuse_sentences(sentences, similarity_threshold, max_cluster_size)

        # Combine fused sentences
        optimized_description = "\n".join(optimized_sentences)
        
        # Print statistics
        original_chars = len(original_description)
        optimized_chars = len(optimized_description)
        reduction = original_chars - optimized_chars
        reduction_percent = (reduction / original_chars) * 100 if original_chars > 0 else 0
        
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
    # result = asyncio.run(optimizer.optimize_single_node(entity_graph, node_name, similarity_threshold=0.9, max_cluster_size=5))
    # print("Optimization complete")

    # If you want to save the result to a file
    # with open(f"./optimized_node_{node_name.replace(' ', '_')}.json", "w") as f:
    #     json.dump(result, f, indent=2)

    # Test QA on the original vs optimized descriptions
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
    optimized_graph = optimizer.optimize_entity_graph(entity_graph)
    
    # Save optimized graph
    optimizer.save_entity_graph(optimized_graph, "./optimized_entity_graph.json")
    optimized_graph = optimizer.load_entity_graph("./optimized_entity_graph.json")
    end_time = time.time()
    print(f"Optimization completed in {end_time - start_time:.2f} seconds")
    # # Print statistics
    original_chars = sum(len(node.get("description", "")) for node_name, node in entity_graph.items())
    optimized_chars = sum(len(node.get("description", "")) for node_name, node in optimized_graph.items())
    print(original_chars)
    print(optimized_chars)
    print(f"Original total characters: {original_chars}")
    print(f"Optimized total characters: {optimized_chars}")
    print(f"Reduction: {original_chars - optimized_chars} characters ({(1 - optimized_chars/original_chars)*100:.2f}%)")