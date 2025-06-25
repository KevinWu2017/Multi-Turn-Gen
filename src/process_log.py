import json
import os

import pydra
from pydra import Config, REQUIRED
from typing import Dict, Optional

from multi_turn_config import MultiTurnConfig

def read_log_file(log_file_path):
    """
    Read the log file and return the parsed JSON data.
    
    Args:
        log_file_path (str): Path to the log file
        
    Returns:
        dict: Parsed JSON data
    """
    try:
        with open(log_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print(f"Error: File '{log_file_path}' not found.")
        return None
    except json.JSONDecodeError as e:
        print(f"Error: Failed to parse JSON - {e}")
        return None

def print_model_responses(log_data):
    """
    Print model responses for each turn.
    
    Args:
        log_data (dict): Parsed log data
    """
    if not log_data:
        print("No data to process.")
        return
    
    # Print metadata first
    if "metadata" in log_data:
        print("=== Metadata ===")
        metadata = log_data["metadata"]
        for key, value in metadata.items():
            print(f"{key}: {value}")
        print()
    
    # Find all turn numbers (exclude metadata)
    turns = [key for key in log_data.keys() if key.isdigit()]
    turns.sort(key=int)  # Sort numerically
    
    if not turns:
        print("No turns found in the log file.")
        return
    
    print("=== Model Responses by Turn ===")
    for turn in turns:
        turn_data = log_data[turn]
        model_reasoning_response = turn_data.get("model_reasoning_response", "")
        model_response = turn_data.get("model_response", "")
        
        print(f"Turn {turn}:")
        if model_reasoning_response:
            print("  [Reasoning]")
            for line in model_reasoning_response.split('\n'):
                print(f"  {line}")
        else:
            print("  (No reasoning response)")
        if model_response:
            # print(f"  Formatted Response:")
            # Print with proper formatting
            print("  [Content]")
            for line in model_response.split('\n'):
                print(f"  {line}")
        else:
            print("  (Empty response)")
        print()

@pydra.main(base=MultiTurnConfig)
def main(
    config: MultiTurnConfig
):
    log_dir = os.path.join(config.log_dir_prefix, config.run_group, config.run_name, "problem_" + str(config.problem_id), "sample_" + str(0))
    log_file_path = log_dir + "/log.json"
    
    print(f"Reading log file: {log_file_path}")
    print("=" * 50)
    
    # Read and process the log file
    log_data = read_log_file(log_file_path)
    print_model_responses(log_data)

if __name__ == "__main__":
    main()