from dotenv import load_dotenv, find_dotenv
from collections import deque
from typing import Dict, List
from fuzzywuzzy import fuzz, process
import json, heapq, re, asyncio, logging
from pydantic import BaseModel
from typing import Dict, List, TypedDict, Annotated, Sequence
from langchain_openai import ChatOpenAI
from langchain.tools import StructuredTool
from langchain_core.prompts import PromptTemplate
from langchain.callbacks.tracers.langchain import wait_for_all_tracers
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.graph import END, StateGraph
from langchain_core.messages import (
    BaseMessage,
    ToolMessage,
    AIMessage,
)

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

class FlowOperations:
    def __init__(self):
        env_path = find_dotenv()
        load_dotenv()
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
        self.replaced_term = {}

        try:
            self.llm.invoke("Hello, World!")
        finally:
            wait_for_all_tracers()

    # === Agents Functions ===
    def set_replaced_term(self, entity_name: str, validated_entity_name: str):
        self.replaced_term[entity_name] = validated_entity_name

    def normalize_entity_name(self, name: str) -> str:
        """
        Normalizes entity names for comparison.
        
        Args:
            name (str): Entity name to normalize
        
        Returns:
            str: Normalized entity name
        """
        return name.strip().upper()

    def load_entity_graph(self, file_path: str) -> Dict:
        """
        Loads the entity graph from a JSON file.

        Args:
            file_path (str): Path to the JSON file containing the entity graph.

        Returns:
            Dict: The loaded entity graph dictionary.
        """
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading entity graph: {e}")
            return {}

    def bfs_traversal(self, graph: Dict, start_entity: str, max_distance: int) -> Dict[str, int]:
        """
        Performs BFS traversal on the graph to compute distances of all reachable entities from the start entity.

        Args:
            graph (Dict): The entity graph dictionary.
            start_entity (str): The name of the entity to start the traversal from.
            max_distance (int): The maximum distance to traverse.

        Returns:
            Dict[str, int]: A dictionary mapping entity names to their distances from the start entity.
        """
        if start_entity not in graph:
            all_entities = list(entity.upper() for entity in graph.keys())
            best_matches = process.extract(start_entity, all_entities, scorer=fuzz.ratio, limit=1)
            best_match, best_score = best_matches[0]
            print("This is fuzzy matching result: ", best_match, best_score)
            if best_score > 95:
                start_entity = best_match
            else:
                print(f"Entity '{start_entity}' not found in the graph.")
                return {}

        distances = {start_entity: 0}
        queue = deque([(start_entity, 0)])
        visited = set()

        while queue:
            current_entity, current_distance = queue.popleft()
            if current_entity in visited:
                continue
            visited.add(current_entity)
            distances[current_entity] = current_distance
            if current_distance >= max_distance:
                continue

            connections = graph.get(current_entity, {}).get('connections', [])
            for connection in connections:
                neighbor = connection.get('target')
                queue.append((neighbor, current_distance + 1))

        return distances

    def get_subgraph(self, graph: Dict, start_node: str, max_distance: int) -> Dict:
        """
        Extracts a subgraph containing nodes within a specified distance from the start node.

        Args:
            graph (Dict): The complete entity graph.
            start_node (str): The node from which to calculate distances.
            max_distance (int): The maximum distance from the start node.

        Returns:
            Dict: Subgraph with nodes and edges within the specified distance.
        """
        distances = self.bfs_traversal(graph, start_node, max_distance)
        if not distances:
            return {}

        subgraph = {}
        for node in distances.keys():
            node_info = {
                'type': graph[node]['type'],
                'description': graph[node]['description'],
                'distance': distances[node],
                'connections': []
            }

            # Only include connections if the distance of the node is less than max_distance
            if distances[node] < max_distance:
                node_info['connections'] = [
                    conn for conn in graph[node]['connections'] if conn['target'] in distances
                ]

            subgraph[node] = node_info

        return subgraph
    
    def extract_json(self, text):
        """
        Extract JSON content from a string that may contain Markdown code block delimiters.

        Args:
            text (str): The response text from the LLM.

        Returns:
            dict or None: Parsed JSON object if extraction and parsing are successful; otherwise, None.
        """
        # Define a regex pattern to extract JSON within ```json ... ```
        json_pattern = r'```json\s*(\{.*\})\s*```'
        match = re.search(json_pattern, text, re.DOTALL)
        if match:
            json_str = match.group(1)
            logging.debug(f"Extracted JSON string: {json_str}")
            try:
                data = json.loads(json_str)
                logging.debug("JSON successfully parsed into dictionary.")
                return data
            except json.JSONDecodeError as e:
                logging.error(f"JSON decoding error: {e}")
                return None
        else:
            logging.error("No JSON found in the response.")
            return None
    
    def build_entity_list(self, subgraph: Dict) -> Dict[str, Dict]:
        """
        Builds an entity list by traversing the subgraph in ascending order of distance.

        Args:
            subgraph (Dict): The subgraph containing entities and their connections.

        Returns:
            Dict[str, Dict]: A dictionary where each key is an entity name and its value contains
                            'relationship', 'description', 'type', and 'distance'.
        """
        entity_list = {}
        heap = []
        processed_connections = set()

        # Initialize the heap with entities sorted by distance
        for entity, attrs in subgraph.items():
            distance = attrs.get('distance', 0)
            heapq.heappush(heap, (distance, entity))

        while heap:
            distance, entity = heapq.heappop(heap)

            if entity in entity_list:
                continue  # Skip if already processed

            attrs = subgraph.get(entity, {})
            entity_info = {
                'relationship': [],
                'description': attrs.get('description', ''),
                'type': attrs.get('type', 'UNKNOWN'),
                'distance': distance
            }
            entity_list[entity] = entity_info

            connections = attrs.get('connections', [])

            for connection in connections:
                target = connection.get('target')
                description = connection.get('description', '')

                # Create a frozenset to handle undirectional relationships
                connection_key = frozenset([entity, target])

                if connection_key not in processed_connections:
                    # Add relationship description to the current entity
                    entity_list[entity]['relationship'].append(description)

                    # Mark this connection as processed
                    processed_connections.add(connection_key)

        # Optionally, convert relationship lists to concatenated strings
        for entity, info in entity_list.items():
            info['relationship'] = " ".join(info['relationship'])

        return entity_list
    
    def find_common_entities(self, entity_lists: Dict) -> tuple[Dict[str, Dict], set[str]]:
        """
        Find common entities across different entity lists where their shortest distance isn't 0.
        
        Args:
            entity_lists (Dict): Dictionary containing multiple entity lists with their distances
            
        Returns:
            Tuple[Dict, set]: 
                - Dictionary of common entities with their data (using shortest distance)
                - Set of common entity names for filtering
        """
        # First, collect all entities and their data across lists
        entity_data = {}  # {entity: {list_id: {distance: X, ...other_data}}}
        
        for list_id, entities in entity_lists.items():
            for entity_name, entity_info in entities.items():
                if entity_name not in entity_data:
                    entity_data[entity_name] = {}
                entity_data[entity_name][list_id] = entity_info
        
        # Find common entities and their data
        common_entities = {}
        for entity, appearances in entity_data.items():
            if len(appearances) > 1:  # Appears in multiple lists
                # Get shortest distance across all appearances
                min_distance = min(data.get('distance', float('inf')) 
                                for data in appearances.values())
                
                if min_distance > 0:  # Only include if shortest distance > 0
                    # Get the data from the appearance with the shortest distance
                    shortest_appearance = min(
                        appearances.items(),
                        key=lambda x: x[1].get('distance', float('inf'))
                    )
                    common_entities[entity] = shortest_appearance[1]
        
        return common_entities, set(common_entities.keys())

    def process_tool_messages(self, messages) -> Dict:
        """
        Process tool messages and aggregate their responses into entity lists.
        
        Args:
            messages (list): List of messages to process
            
        Returns:
            Dict: Dictionary containing either entity lists or error message
        """
        entity_lists = {}
        acc = 0
        
        for msg in reversed(messages):
            if not isinstance(msg, ToolMessage):
                break
                
            try:
                response = json.loads(msg.content)
                if "messages" in response:
                    no_entity_msg = response["messages"][0]
                    if "No matching entity found" in no_entity_msg:
                        return {"messages": AIMessage(content=no_entity_msg)}
                        
                entity_lists[f"entity_list_{acc}"] = response
                acc += 1
                
            except Exception as e:
                print(f"Error formatting tool response: {e}")
                
        return {"messages": AIMessage(content=json.dumps(entity_lists, indent=2))}
    
    def load_json(self, file_path: str) -> Dict:
        """
        Loads the alias index from JSON file.
        
        Args:
            file_path (str): Path to the alias index JSON file
        
        Returns:
            Dict: The loaded alias index dictionary
        """
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading alias index: {e}")
            return {}
        
    def ask_user(self, **kwargs):
        # case 1: confirm fuzzy match
        if 'match_name' in kwargs and 'original_name' in kwargs:
            while True:
                print(f"Did you mean '{kwargs['match_name']}' when you entered '{kwargs['original_name']}'? (Y/N)")
                user_input = input()
                if user_input.upper() in ['Y', 'N']:
                    return kwargs['match_name'] if user_input.upper() == 'Y' else None
                print("Please enter Y or N")
        
        # case 2: multiple full names for abbreviation
        if 'alias_names' in kwargs:
            while True:
                print("Multiple options found. Please choose one:")
                for i, name in enumerate(kwargs['alias_names'], 1):
                    print(f"{i}. {name}")
                try:
                    choice = int(input("Enter the number of your choice: "))
                    if 1 <= choice <= len(kwargs['alias_names']):
                        return kwargs['alias_names'][choice - 1]
                except ValueError:
                    print("Please enter a valid number")
        
        return None

    def check_entity_in_graph(self, entity_name):
        entity_graph = self.load_json('./optimized_entity_graph.json')
        return entity_name in entity_graph.keys()

    def fuzzy_match_entity(self, entity_name, threshold=95) -> str:
        """
        Perform fuzzy matching on all node names in the graph.
        
        Args:
            entity_name (str): The entity name to match
            threshold (int): The minimum score to consider a match
            
        Returns:
            str: The matched entity name or None if no match above threshold
        """
        graph_entities = list(self.load_json('./optimized_entity_graph.json').keys())
            
        matches = process.extract(entity_name, graph_entities, scorer=fuzz.ratio, limit=1)
        
        if matches and matches[0][1] > threshold:
            return matches[0][0]
        
        return None

    def match_entity_name_with_alias(self, entity_name: str) -> str:
        """
        Matches an entity name to the closest entity in the entity graph using the comprehensive logic.
        
        Args:
            query (str): The entity name to match
            
        Returns:
            str: Validated entity name
        """
        
        alias_dict= self.load_json('./alias_dict.json')
        # STEP 1: Determine if query is an abbreviation or full name
        is_abbreviation = entity_name in alias_dict.get('abbreviations', {})
        
        # STEP 2: Process as abbreviation
        if is_abbreviation:
            full_names = alias_dict['abbreviations'].get(entity_name, [])
            # Check if abbreviation exists directly in graph
            if self.check_entity_in_graph(entity_name):
                if len(full_names) > 1:
                    chosen_name = self.ask_user(alias_names=full_names)
                    self.set_replaced_term(entity_name, chosen_name)
                elif len(full_names) == 1:
                    self.set_replaced_term(entity_name, full_names[0])
                return entity_name
            
            # Filter to only full names that exist in the graph
            valid_full_names = [name for name in full_names if self.check_entity_in_graph(self.normalize_entity_name(name))]
            
            if len(valid_full_names) > 1:
                # Ask user to choose from multiple valid full names
                chosen_name = self.ask_user(alias_names=valid_full_names)
                if chosen_name:
                    self.set_replaced_term(entity_name, chosen_name)
                    return self.normalize_entity_name(chosen_name)
            elif len(valid_full_names) == 1:
                self.set_replaced_term(entity_name, valid_full_names[0])
                return self.normalize_entity_name(valid_full_names[0])
        
        # STEP 3: Process as full name
        else:
            # Check if full name exists directly in graph
            if self.check_entity_in_graph(entity_name):
                return entity_name
            
            for abbr, full_names in alias_dict['abbreviations'].items():
                if entity_name in [name.strip().upper() for name in full_names]:
                    # Check if the abbreviation exists in the graph
                    if self.check_entity_in_graph(abbr):
                        return abbr
                    
                    # Filter to only full names that exist in the graph
                    valid_full_names = [name for name in full_names if self.check_entity_in_graph(self.normalize_entity_name(name))]
                    
                    if len(valid_full_names) > 1:
                        # Ask user to choose from multiple valid full names
                        chosen_name = self.ask_user(alias_names=valid_full_names)
                        if chosen_name:
                            self.set_replaced_term(entity_name, chosen_name)
                            return self.normalize_entity_name(chosen_name)
                    elif len(valid_full_names) == 1:
                        # Only one valid full name exists
                        self.set_replaced_term(entity_name, valid_full_names[0])
                        return self.normalize_entity_name(valid_full_names[0])
        
        # STEP 4: Fuzzy matching as fallback
        match_name = self.fuzzy_match_entity(entity_name)
        
        if match_name:
            # Ask user to confirm the fuzzy match
            confirmed_name = self.ask_user(match_name=match_name, original_name=entity_name)
            if confirmed_name:
                self.set_replaced_term(entity_name, confirmed_name)
                return entity_name, self.normalize_entity_name(confirmed_name)
        
        # No match found
        return "None"
    
    def split_chunks_by_candidate(self, background_chunks: str, group_size: int = 5) -> List[str]:
        """
        Split background chunks by candidate answer sections.
        
        Args:
            background_chunks (str): The full background text containing multiple candidate answers
            
        Returns:
            List[str]: List of individual candidate answer segments
        """
        # Split by the candidate answer delimiter
        individual_chunks = background_chunks.split("-------------------\n")
        individual_chunks = [chunk.strip() for chunk in individual_chunks if chunk.strip()]
        
        # Group chunks into segments of specified size
        grouped_segments = []
        for i in range(0, len(individual_chunks), group_size):
            group = individual_chunks[i:i + group_size]
            # Rejoin the group with the delimiter
            segment = "\n-------------------\n".join(group)
            grouped_segments.append(segment)
        
        return grouped_segments

    async def process_entity_list(self, entities, background_chunks, llm):
        """
        Process a single entity list asynchronously with candidate answer splitting.
        """
        try:
            entities_dict = entities
            entity_names = list(entities_dict.keys())
            
            # Split background chunks into candidate answers
            candidate_segments = self.split_chunks_by_candidate(background_chunks, group_size=5)
            
            prompt_template = """
            You are responsible for extracting related information about each entity from the provided candidate answer.
            Only extract information if it's directly relevant to the entities.
            
            For each entity name provided, find and extract any relevant information from this candidate answer.
            If no relevant information is found for an entity, exclude it from the output.
            
            Entities:
            {entity_names}
            
            Candidate Answer:
            {background_chunks}
            
            Example Output:
            {{
                "ENTITY_1": "Related info from this candidate answer",
                "ENTITY_2": "Related info from this candidate answer"
            }}
            """
            
            # Create tasks for processing each candidate segment
            tasks = []
            for segment in candidate_segments:
                prompt = PromptTemplate.from_template(prompt_template)
                chain = prompt | llm
                task = chain.ainvoke({
                    "entity_names": json.dumps(entity_names),
                    "background_chunks": segment
                })
                tasks.append(task)
            
            # Process all segments concurrently
            responses = await asyncio.gather(*tasks)
            
            # Merge information from all segments
            merged_info = {}
            for response in responses:
                related_info = self.extract_json(response.content)
                if isinstance(related_info, dict):
                    for entity, info in related_info.items():
                        if entity not in merged_info:
                            merged_info[entity] = []
                        merged_info[entity].append(info)
            
            # Update entities_dict with merged information
            for entity, info_list in merged_info.items():
                if entity in entities_dict:
                    entities_dict[entity]['chunk_context'] = '\n'.join(info_list)
            print("This is the entities_dict:\n", entities_dict)
            return entities_dict
            
        except Exception as e:
            print(f"Error processing entity list: {e}")
            return None
        
class FlowConstructor:
    def __init__(self):
        self.flow_operations = FlowOperations()
        self.find_entities_tool = StructuredTool.from_function(
            func=self.find_entities,
            name="find_entities",
            description="Find related entities in the graph for a given entity name.",
            return_direct=False
        )
        self.graph_path = './optimized_entity_graph.json'
        self.subgraph_distance = 2
    
    def set_subgraph_distance(self, distance: int):
        self.subgraph_distance = distance

    def find_entities(self, entity_name: str) -> Dict:
        """
        Find related entities in the graph for a given entity name.
        
        Args:
            entity_name (str): Name of the entity to search for
        
        Returns:
            Dict: Related entities grouped by distance
        """
        # Normalize the input entity name
        normalized_input = self.flow_operations.normalize_entity_name(entity_name)
        entity_graph = self.flow_operations.load_entity_graph(self.graph_path)
        validated_entity_name = self.flow_operations.match_entity_name_with_alias(normalized_input)
        if validated_entity_name:
            subgraph = self.flow_operations.get_subgraph(entity_graph, validated_entity_name, self.subgraph_distance)
            return self.flow_operations.build_entity_list(subgraph)
        else:
            return {"messages": [f"No matching entity found for '{entity_name}'"]}

    def entity_extractor(self, state: AgentState) -> Dict:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        tools = [self.find_entities_tool]
        llm_with_tools = llm.bind_tools(tools)
        
        # Check if we have a tool message
        last_message = state["messages"][-1]
        if isinstance(last_message, ToolMessage):
            # Parse the tool results directly and format them
            return self.flow_operations.process_tool_messages(state["messages"])
        
        # If not a tool message, proceed with initial query
        prompt = PromptTemplate.from_template("""
        You are an expert at identifying and extracting key entities from questions.
        Identify ALL main entities from the question and use the find_entities tool for each one.
        Input question: {message}
        """)
        
        chain = prompt | llm_with_tools
        full_content = state["messages"][0].content
        question = full_content.split("Background Chunks:", 1)[0].replace("Question:", "").strip() if "Background Chunks:" in full_content else full_content
        response = chain.invoke({"message": question})
        return {"messages": response}

    async def chunk_builder(self, state: AgentState) -> Dict:
        """
        Refactored chunk_builder to handle large entity lists by processing entity names separately
        and ensuring each entity's related information is a single string.
        
        Args:
            state (AgentState): The current state of the agent.
        
        Returns:
            Dict: Updated messages with merged entities information in the format {"Entity Name": "related info", ...}
        """
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

        # Step 1: Extract Background Chunks and Entities
        message = state["messages"]
        last_message = message[-1]
        full_content = message[0].content
        background_chunks = (
            full_content.split("Background Chunks:", 1)[1].strip()
            if "Background Chunks:" in full_content
            else full_content
        )
        content_str = last_message.content.strip()
        entity_lists = json.loads(content_str)
        # Create tasks for parallel processing
        tasks = []
        for entity_list, entities in entity_lists.items():
            task = self.flow_operations.process_entity_list(entities, background_chunks, llm)
            tasks.append(task)
        
        # Run tasks concurrently
        entities_chunks = {}
        results = await asyncio.gather(*tasks)
        
        # Process results
        for i, result in enumerate(results, 1):
            if result:
                entities_chunks[f"entity_chunk_{i}"] = result
        
        # Save results
        res = json.dumps(entities_chunks, indent=4)
        with open('entity_chunks.json', 'w') as f:
            f.write(res)
        
        return {"messages": AIMessage(content=res)}
    
    def router(self, state):
        # This is the router
        messages = state["messages"]
        last_message = messages[-1]
        # If the message is already formatted as a dictionary, move to chunk builder
        if isinstance(last_message.content, dict) or last_message.content.startswith("{"):
            return "continue"
            
        # If there are tool calls, handle them
        if last_message.tool_calls:
            return "call_tool"
            
        no_match_keywords = [
            "No matching entity found",
            "No entities found for your query",
            "No entities were identified",
            "Please try rephrasing your question or provide more context"
        ]
        
        if any(keyword in last_message.content for keyword in no_match_keywords):
            return "end"

        return "continue"
    
    def create_agentic_flow(self, **kwargs):
        if "subgraph_distance" in kwargs:
            self.set_subgraph_distance(kwargs["subgraph_distance"])
        workflow = StateGraph(AgentState)
        workflow.add_node("EntityExtractor", self.entity_extractor)
        workflow.add_node("ChunkBuilder", self.chunk_builder)

        tools = [self.find_entities]
        tool_node = ToolNode(tools)
        workflow.add_node("tools", tool_node)

        # workflow.add_edge("EntityExtractor", "ChunkBuilder")
        workflow.add_edge("ChunkBuilder", END)
        workflow.add_conditional_edges(
            "EntityExtractor",
            self.router,
            path_map={"call_tool": "tools", "continue": "ChunkBuilder", "end": END}
        )
        workflow.add_edge("tools", "EntityExtractor")
        workflow.set_entry_point("EntityExtractor")


        agentic_flow = workflow.compile()
        return agentic_flow