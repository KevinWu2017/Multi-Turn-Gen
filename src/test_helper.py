import os
import json
REPO_TOP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dir = os.path.join(REPO_TOP_DIR, "results/log_for_unit_test")

log_for_unit_test_path = {
    "none": os.path.join(dir, "Success.json"),
    "compile": os.path.join(dir, "Compile_Error_CPU.json"),
    "correctness": os.path.join(dir, "Correctness_Error.json"),
    "cuda_error": os.path.join(dir, "CUDA_Error.json"),
}

def get_log_for_unit_test_path(simulate_error: str) -> str:
    log_file = log_for_unit_test_path.get(simulate_error, None)
    if log_file and os.path.exists(log_file):
        with open(log_file, 'r') as f:
            current_log = json.load(f)

            model_reasoning_response = current_log['1'].get("model_reasoning_response", "")
            model_response = current_log['1'].get("model_response", "")

            return model_reasoning_response, model_response
    else:
        raise FileNotFoundError(f"Log file for simulate_error '{simulate_error}' not found at {log_file}")
