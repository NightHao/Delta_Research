import pandas as pd
import json

# Path to your GraphRAG data
base_path = "/home/yeskey525/Research_CODE/graph_rag_default_auto_prompt_tuning/ragtest/output/"

# Load entities and relationships
entity_df = pd.read_parquet(f"{base_path}create_final_nodes.parquet")
relation_df = pd.read_parquet(f"{base_path}create_final_relationships.parquet")

# Format data for entities_to_graph.py
results = []

# Add entities
for _, row in entity_df.iterrows():
    entity_record = {
        "response": f'"entity"<|>{row["title"]}<|>{row["type"]}<|>{row["description"]}##'
    }
    results.append(entity_record)

# Add relationships
for _, row in relation_df.iterrows():
    description = row.get("description", "")
    if pd.isna(description):
        description = f"Relationship between {row['source']} and {row['target']}"
        
    relationship_record = {
        "response": f'"relationship"<|>{row["source"]}<|>{row["target"]}<|>{description}<|>{row["weight"]}##'
    }
    results.append(relationship_record)

# Save to JSON
with open("entities_result.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=4)

print(f"Processed {len(entity_df)} entities and {len(relation_df)} relationships")