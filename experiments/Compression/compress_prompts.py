#!/usr/bin/env python3
"""
Prompt Compression Script using LLMLingua

This script compresses prompts in a JSON file using Microsoft's LLMLingua
and saves the compressed prompts to a new JSON file.
"""

import json
import os
import argparse
from tqdm import tqdm
import torch

try:
    from llmlingua import PromptCompressor
except ImportError:
    print("LLMLingua not found. Installing...")
    import subprocess
    subprocess.check_call(["pip", "install", "llmlingua"])
    from llmlingua import PromptCompressor


def load_json_from_file(file_path):
    """Load JSON data from a file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def find_compressible_strings(obj, path="", min_length=200):
    """Find all string fields in a nested structure that should be compressed."""
    paths = []
    
    if isinstance(obj, dict):
        for key, value in obj.items():
            new_path = f"{path}.{key}" if path else key
            if isinstance(value, str) and len(value) >= min_length:
                paths.append((new_path, value))
            elif isinstance(value, (dict, list)):
                paths.extend(find_compressible_strings(value, new_path, min_length))
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            new_path = f"{path}[{i}]"
            if isinstance(item, str) and len(item) >= min_length:
                paths.append((new_path, item))
            elif isinstance(item, (dict, list)):
                paths.extend(find_compressible_strings(item, new_path, min_length))
                
    return paths


def set_value_at_path(obj, path, value):
    """Set a value at a specific path in a nested structure."""
    if '.' in path:
        key, rest = path.split('.', 1)
        if '[' in key and key.endswith(']'):
            list_key, idx = key.split('[', 1)
            idx = int(idx[:-1])
            set_value_at_path(obj[list_key][idx], rest, value)
        else:
            set_value_at_path(obj[key], rest, value)
    elif '[' in path and path.endswith(']'):
        key, idx = path.split('[', 1)
        idx = int(idx[:-1])
        obj[key][idx] = value
    else:
        obj[path] = value


def process_data_in_batches(data, llm_lingua, compression_params, batch_size=10):
    """Process the data in small batches to avoid memory issues."""
    compressed_data = []
    total_items = len(data)
    total_batches = (total_items + batch_size - 1) // batch_size
    
    total_original_chars = 0
    total_compressed_chars = 0
    
    for batch_idx in tqdm(range(total_batches), desc="Processing batches"):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, total_items)
        batch = data[start_idx:end_idx]
        
        batch_compressed = []
        for item in batch:
            # Create a copy of the item to modify
            compressed_item = item.copy()
            
            # Find all compressible strings in the item
            compressible_strings = find_compressible_strings(item)
            
            for path, text in compressible_strings:
                try:
                    # Compress the text
                    result = llm_lingua.compress_prompt(text, **compression_params)
                    compressed_text = result['compressed_prompt']
                    
                    # Update statistics
                    total_original_chars += len(text)
                    total_compressed_chars += len(compressed_text)
                    
                    # Set the compressed text in the item
                    set_value_at_path(compressed_item, path, compressed_text)
                except Exception as e:
                    print(f"Error compressing text at path {path}: {e}")
            
            batch_compressed.append(compressed_item)
        
        compressed_data.extend(batch_compressed)
        
    # Calculate overall compression ratio
    compression_ratio = total_compressed_chars / total_original_chars if total_original_chars > 0 else 1.0
    print(f"\nOverall compression ratio: {compression_ratio:.2f}")
    print(f"Original text size: {total_original_chars:,} chars")
    print(f"Compressed text size: {total_compressed_chars:,} chars")
    print(f"Characters saved: {total_original_chars - total_compressed_chars:,} chars")
    
    return compressed_data


def main():
    """Main function to compress prompts."""
    parser = argparse.ArgumentParser(description='Compress prompts in a JSON file using LLMLingua')
    parser.add_argument('--input', default='final_prompt.json', help='Input JSON file path')
    parser.add_argument('--output', default='fixed_compressed_final_prompt.json', help='Output JSON file path')
    parser.add_argument('--rate', type=float, default=0.5, help='Compression rate (0.0-1.0)')
    parser.add_argument('--batch-size', type=int, default=10, help='Batch size for processing')
    parser.add_argument('--min-length', type=int, default=200, help='Minimum text length to compress')
    parser.add_argument('--use-large-model', action='store_true', help='Use the large model instead of base model')
    
    args = parser.parse_args()
    
    # Check if CUDA is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Initialize LLMLingua model
    if args.use_large_model:
        model_name = "microsoft/llmlingua-2-xlm-roberta-large-meetingbank"
        print("Using large model (requires more GPU memory)")
    else:
        model_name = "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank"
        print("Using base model")
    
    llm_lingua = PromptCompressor(
        model_name=model_name,
        use_llmlingua2=True,
        device=device
    )
    
    # Configure compression parameters
    compression_params = {
        "rate": args.rate,
        "force_tokens": ['\n', '?'],  # Preserve newlines and question marks
    }
    
    try:
        # Load the prompt data
        print(f"Loading prompts from {args.input}...")
        data = load_json_from_file(args.input)
        print(f"Loaded {len(data)} prompts successfully")
        
        # Process the data
        compressed_data = process_data_in_batches(
            data, 
            llm_lingua, 
            compression_params, 
            batch_size=args.batch_size
        )
        
        # Save the compressed data
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(compressed_data, f, ensure_ascii=False, indent=2)
        
        print(f"\nCompressed data saved to {args.output}")
        print(f"File size: {os.path.getsize(args.output) / (1024*1024):.2f} MB")
        
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main() 