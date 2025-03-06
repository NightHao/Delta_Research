import json
import networkx as nx
import re
from typing import Dict

def clean_str(input_str: str) -> str:
    """Clean and strip the input string."""
    return input_str.strip()

def _unpack_descriptions(data: dict) -> set[str]:
    """
    Given a node or edge's dictionary, return a set of description lines.
    If a description exists as a multi-line string, split it.
    """
    desc = data.get("description", "")
    return set(desc.split("\n")) if desc else set()

def _unpack_source_ids(data: dict) -> set[str]:
    """
    Given a node or edge's dictionary, return a set of source IDs.
    If the source_id is a comma separated string, split it.
    """
    src = data.get("source_id", "")
    return set(src.split(", ")) if src else set()

def process_results(
    results: Dict[int, str],
    tuple_delimiter: str = "<|>",
    record_delimiter: str = "##",
    join_descriptions: bool = True
) -> nx.Graph:
    """
    Process the results dict (mapping doc_id to response) and build an undirected graph.
    
    Each record in a response may correspond to either an entity (node) or a relationship (edge).
    """
    graph = nx.Graph()
    for source_doc_id, extracted_data in results.items():
        records = [r.strip() for r in extracted_data.split(record_delimiter)]
        for record in records:
            if not record:
                continue
            # Remove any wrapping parentheses
            record = re.sub(r"^\(|\)$", "", record.strip())
            record_attributes = record.split(tuple_delimiter)
            if not record_attributes or len(record_attributes) < 1:
                continue
            # Process entity record
            if record_attributes[0] == '"entity"' and len(record_attributes) >= 4:
                entity_name = clean_str(record_attributes[1].upper())
                entity_type = clean_str(record_attributes[2].upper())
                entity_description = clean_str(record_attributes[3])
                
                if entity_name in graph.nodes():
                    node = graph.nodes[entity_name]
                    if join_descriptions:
                        # Merge the descriptions (using a set to avoid duplicates)
                        all_desc = _unpack_descriptions(node)
                        all_desc.add(entity_description)
                        node["description"] = "\n".join(all_desc)
                    else:
                        # Keep the longer description if not merging
                        if len(entity_description) > len(node.get("description", "")):
                            node["description"] = entity_description
                    new_sources = _unpack_source_ids(node)
                    new_sources.add(str(source_doc_id))
                    node["source_id"] = ", ".join(new_sources)
                    # Set type if not already set
                    if not node.get("type"):
                        node["type"] = entity_type
                else:
                    graph.add_node(
                        entity_name,
                        type=entity_type,
                        description=entity_description,
                        source_id=str(source_doc_id)
                    )

            # Process relationship record
            elif record_attributes[0] == '"relationship"' and len(record_attributes) >= 5:
                source = clean_str(record_attributes[1].upper())
                target = clean_str(record_attributes[2].upper())
                edge_description = clean_str(record_attributes[3])
                edge_source_id = clean_str(str(source_doc_id))
                try:
                    weight = float(record_attributes[-1])
                except ValueError:
                    weight = 1.0

                # Ensure both nodes exist.
                if source not in graph.nodes():
                    graph.add_node(source, type="", description="", source_id=edge_source_id)
                if target not in graph.nodes():
                    graph.add_node(target, type="", description="", source_id=edge_source_id)
                if graph.has_edge(source, target):
                    edge_data = graph.get_edge_data(source, target)
                    if edge_data is not None:
                        weight += edge_data.get("weight", 1.0)
                        if join_descriptions:
                            existing_desc = _unpack_descriptions(edge_data)
                            existing_desc.add(edge_description)
                            edge_description = "\n".join(existing_desc)
                        new_sources = _unpack_source_ids(edge_data)
                        new_sources.add(str(source_doc_id))
                        edge_source_id = ", ".join(new_sources)
                graph.add_edge(
                    source,
                    target,
                    weight=weight,
                    description=edge_description,
                    source_id=edge_source_id
                )
    return graph

def load_results(json_path: str) -> Dict[int, str]:
    """
    Load the JSON file containing an array of objects, each
    expected to have a "response" key.
    Returns a dictionary mapping an index (as document id) to the response string.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    results = {}
    for idx, item in enumerate(data):
        if "response" in item:
            results[idx] = item["response"]
    return results

def convert_graph_to_dict(g: nx.Graph) -> Dict[str, dict]:
    """
    Convert the NetworkX graph to a dictionary representation.
    
    The resulting dictionary has node names as keys. Each node's value is a dict 
    containing its type, description, and a list of connections.
    
    Each connection is represented as a dict with keys: target, weight, and description.
    """
    entity_graph = {}
    for node, data in g.nodes(data=True):
        # Build node entry with type and description
        node_entry = {
            "type": data.get("type", ""),
            "description": data.get("description", ""),
            "connections": []
        }
        # Add all connected edges
        for neighbor in g.neighbors(node):
            edge_data = g.get_edge_data(node, neighbor)
            connection = {
                "target": neighbor,
                "weight": edge_data.get("weight", 1.0),
                "description": edge_data.get("description", "")
            }
            node_entry["connections"].append(connection)
        entity_graph[node] = node_entry
    return entity_graph

if __name__ == "__main__":
    json_path = "./entities_result.json"
    results = load_results(json_path)

    # Process the results to construct a NetworkX graph.
    g = process_results(results, tuple_delimiter="<|>", record_delimiter="##", join_descriptions=True)

    # Convert the NetworkX graph into a dictionary.
    entity_graph = convert_graph_to_dict(g)

    # Write the graph dictionary to a JSON file.
    output_path = "./entity_graph.json"
    with open(output_path, "w", encoding="utf-8") as outfile:
        json.dump(entity_graph, outfile, indent=4, ensure_ascii=False)
    print(f"Entity graph written to {output_path}")