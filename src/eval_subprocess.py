import os
import sys
import subprocess
import json
import tempfile
import time
import shutil
from typing import Dict, Any, Optional
from pydra import Config
import torch

# Add KernelBench to path for imports
KERNELBENCH_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "third_party", "KernelBench")
if KERNELBENCH_PATH not in sys.path:
    sys.path.insert(0, KERNELBENCH_PATH)
from src import eval as kernel_eval
from src import utils as kernel_utils
from utils import timeout

"""
Subprocess-based evaluation code for running torch operations in isolated processes
"""

def get_kernel_hash(kernel_src: str) -> str:
    return str(hash(kernel_src))

def create_eval_script_content(ref_arch_src: str, kernel_src: str, configs: Dict[str, Any], 
                              build_dir: str, device_id: int) -> str:
    """
    Create the content of the evaluation script that will be run in subprocess
    """
    script_content = f'''
import sys
import os
import json
import torch
import time

# Add KernelBench to path
KERNELBENCH_PATH = "{KERNELBENCH_PATH}"
if KERNELBENCH_PATH not in sys.path:
    sys.path.insert(0, KERNELBENCH_PATH)

from src import eval as kernel_eval
from src import utils as kernel_utils

def main():
    try:
        # Set GPU architecture
        kernel_utils.set_gpu_arch({configs.get("gpu_arch", ["Ada"])})
        
        # Create device
        device = torch.device(f"cuda:{device_id}")
        
        # Define source codes
        ref_arch_src = """{ref_arch_src}"""
        
        kernel_src = """{kernel_src}"""
        
        # Run evaluation
        eval_result = kernel_eval.eval_kernel_against_ref(
            original_model_src=ref_arch_src,
            custom_model_src=kernel_src,
            measure_performance={configs.get("measure_performance", False)},
            verbose={configs.get("verbose", False)},
            num_correct_trials={configs.get("num_correct_trials", 1)},
            num_perf_trials={configs.get("num_perf_trials", 10)},
            build_dir="{build_dir}",
            device=device
        )
        
        # Convert result to serializable format
        result_dict = {{
            "compiled": eval_result.compiled,
            "correctness": eval_result.correctness,
            "runtime": eval_result.runtime,
            "metadata": eval_result.metadata
        }}
        
        # Output result as JSON
        print("EVAL_RESULT_START")
        print(json.dumps(result_dict))
        print("EVAL_RESULT_END")
        
    except Exception as e:
        error_dict = {{
            "compiled": False,
            "correctness": False,
            "runtime": None,
            "metadata": {{
                "subprocess_error": str(e),
                "error_type": type(e).__name__
            }}
        }}
        print("EVAL_RESULT_START")
        print(json.dumps(error_dict))
        print("EVAL_RESULT_END")

if __name__ == "__main__":
    main()
'''
    return script_content

def create_profiler_script_content(ref_arch_src: str, kernel_src: str, build_dir: str, 
                                  device_id: int, num_trials: int = 100, 
                                  table_row_limit: int = 10, seed_num: int = 42) -> str:
    """
    Create the content of the profiler script that will be run in subprocess
    """
    script_content = f'''
import sys
import os
import json
import torch
from torch.profiler import profile, record_function, ProfilerActivity

# Add KernelBench to path
KERNELBENCH_PATH = "{KERNELBENCH_PATH}"
if KERNELBENCH_PATH not in sys.path:
    sys.path.insert(0, KERNELBENCH_PATH)

from src import eval as kernel_eval

def main():
    try:
        assert torch.cuda.is_available(), "CUDA is not available, cannot run Torch Profiler"
        
        device = torch.device(f"cuda:{device_id}")
        
        # Define source codes
        ref_arch_src = """{ref_arch_src}"""
        kernel_src = """{kernel_src}"""
        
        kernel_hash = str(hash(kernel_src))
        build_dir = os.path.join("{build_dir}", kernel_hash)
        
        context = {{}}
        _, get_init_inputs, get_inputs = kernel_eval.load_original_model_and_inputs(
            ref_arch_src, context
        )
        
        kernel_eval.set_seed({seed_num})
        inputs = get_inputs()
        init_inputs = get_init_inputs()
        inputs = [
            x.cuda(device=device) if isinstance(x, torch.Tensor) else x
            for x in inputs
        ]
        init_inputs = [
            x.cuda(device=device) if isinstance(x, torch.Tensor) else x
            for x in init_inputs
        ]
        
        ModelNew = kernel_eval.load_custom_model(kernel_src, context, build_dir)
        # construct the new model with init inputs
        model = ModelNew(*init_inputs)
        assert hasattr(model, "forward")
        torch.cuda.synchronize(device=device)
        
        model = model.cuda(device=device)
        
        with torch.no_grad():
            profiling_scheduler = torch.profiler.schedule(
                skip_first=2,
                wait=2,
                warmup=3,
                active={num_trials},
            )
            
            with profile(
                activities=[ProfilerActivity.CUDA],
                schedule=profiling_scheduler,
            ) as prof:
                for _ in range({num_trials}):
                    output = model(*inputs)
                    prof.step()
            
            profiler_output = prof.key_averages().table(sort_by='cuda_time_total', 
                                                        row_limit={table_row_limit})
        
        # Output result
        print("PROFILER_RESULT_START")
        print(profiler_output)
        print("PROFILER_RESULT_END")
        
    except Exception as e:
        print("PROFILER_RESULT_START")
        print(f"Profiler Error: {{str(e)}}")
        print("PROFILER_RESULT_END")

if __name__ == "__main__":
    main()
'''
    return script_content

def compile_single_sample_subprocess(kernel_src: str, config: Config, build_dir: str, 
                                   timeout_seconds: int = 480) -> tuple[int, str, str]:
    """
    CPU Pre compile kernel using subprocess and capture any errors
    """
    kernel_utils.set_gpu_arch(config.gpu_arch)
    
    kernel_hash = get_kernel_hash(kernel_src)
    kernel_build_dir = os.path.join(build_dir, kernel_hash)
    
    # Create a temporary script for compilation
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        script_content = f'''
import sys
import os

# Add KernelBench to path
KERNELBENCH_PATH = "{KERNELBENCH_PATH}"
if KERNELBENCH_PATH not in sys.path:
    sys.path.insert(0, KERNELBENCH_PATH)

from src import eval as kernel_eval

def main():
    try:
        kernel_src = """{kernel_src}"""
        
        returncode, stdout, err = kernel_eval.build_compile_cache_with_capturing(
            custom_model_src=kernel_src,
            verbose={config.verbose},
            build_dir="{kernel_build_dir}",
        )
        
        print(f"COMPILE_RESULT:{{returncode}}:{{stdout}}:{{err}}")
        
    except Exception as e:
        print(f"COMPILE_RESULT:-1:{{str(e)}}:{{str(e)}}")

if __name__ == "__main__":
    main()
'''
        f.write(script_content)
        script_path = f.name
    
    try:
        # Run the script in subprocess
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            timeout=timeout_seconds
        )
        
        # Parse output
        stdout_lines = result.stdout.strip().split('\n')
        for line in stdout_lines:
            if line.startswith("COMPILE_RESULT:"):
                parts = line.split(':', 3)
                if len(parts) >= 4:
                    returncode = int(parts[1])
                    stdout = parts[2]
                    err = parts[3]
                    return returncode, stdout, err
        
        # Fallback if parsing fails
        return result.returncode, result.stdout, result.stderr
        
    except subprocess.TimeoutExpired:
        return -1, f"Compilation timed out after {timeout_seconds} seconds", f"Compilation timed out after {timeout_seconds} seconds"
    except Exception as e:
        return -1, str(e), str(e)
    finally:
        # Clean up temporary script
        try:
            os.unlink(script_path)
        except:
            pass

def evaluate_single_sample_src_subprocess(ref_arch_src: str, kernel_src: str, configs: Config, 
                                        build_dir: str, device: torch.device, 
                                        timeout_seconds: int = 480) -> kernel_eval.KernelExecResult:
    """
    Evaluate a single sample source code using subprocess execution.
    """
    kernel_hash = get_kernel_hash(kernel_src)
    eval_build_dir = os.path.join(build_dir, kernel_hash)
    
    device_id = device.index if device.type == 'cuda' else 0
    
    # Convert config to dict for serialization
    configs_dict = configs.__dict__ if hasattr(configs, '__dict__') else configs
    
    # Create temporary script for evaluation
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        script_content = create_eval_script_content(
            ref_arch_src, kernel_src, configs_dict, eval_build_dir, device_id
        )
        f.write(script_content)
        script_path = f.name
    
    try:
        # Run the evaluation script in subprocess
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(device_id))
        )
        
        # Parse the output to extract the evaluation result
        stdout_lines = result.stdout.strip().split('\n')
        result_started = False
        result_lines = []
        
        for line in stdout_lines:
            if line.strip() == "EVAL_RESULT_START":
                result_started = True
                continue
            elif line.strip() == "EVAL_RESULT_END":
                break
            elif result_started:
                result_lines.append(line)
        
        if result_lines:
            try:
                result_json = json.loads('\n'.join(result_lines))
                return kernel_eval.KernelExecResult(
                    compiled=result_json["compiled"],
                    correctness=result_json["correctness"],
                    runtime=result_json.get("runtime"),
                    metadata=result_json.get("metadata", {})
                )
            except json.JSONDecodeError as e:
                metadata = {
                    "subprocess_json_error": str(e),
                    "raw_output": result.stdout,
                    "stderr": result.stderr
                }
                return kernel_eval.KernelExecResult(
                    compiled=False, correctness=False, metadata=metadata
                )
        else:
            metadata = {
                "subprocess_no_result": "No evaluation result found in subprocess output",
                "raw_output": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode
            }
            return kernel_eval.KernelExecResult(
                compiled=False, correctness=False, metadata=metadata
            )
            
    except subprocess.TimeoutExpired:
        metadata = {
            "timeout_error": f"Evaluation timed out after {timeout_seconds} seconds",
            "hardware": torch.cuda.get_device_name(device=device) if torch.cuda.is_available() else "unknown",
            "device": str(device)
        }
        return kernel_eval.KernelExecResult(compiled=False, correctness=False, metadata=metadata)
    except Exception as e:
        metadata = {
            "subprocess_error": str(e),
            "hardware": torch.cuda.get_device_name(device=device) if torch.cuda.is_available() else "unknown",
            "device": str(device)
        }
        return kernel_eval.KernelExecResult(compiled=False, correctness=False, metadata=metadata)
    finally:
        # Clean up temporary script
        try:
            os.unlink(script_path)
        except:
            pass

def get_torch_profiler_info_subprocess(ref_arch_src: str, kernel_src: str, build_dir: str, 
                                     device: torch.device, num_trials: int = 100,
                                     table_row_limit: int = 10, seed_num: int = 42) -> str:
    """
    Get the profiler info for a particular kernel using subprocess execution.
    """
    assert torch.cuda.is_available(), "CUDA is not available, cannot run Torch Profiler"
    
    device_id = device.index if device.type == 'cuda' else 0
    
    # Create temporary script for profiling
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        script_content = create_profiler_script_content(
            ref_arch_src, kernel_src, build_dir, device_id, 
            num_trials, table_row_limit, seed_num
        )
        f.write(script_content)
        script_path = f.name
    
    try:
        # Run the profiler script in subprocess
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            timeout=600,  # 10 minutes timeout for profiling
            env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(device_id))
        )
        
        # Parse the output to extract the profiler result
        stdout_lines = result.stdout.strip().split('\n')
        result_started = False
        result_lines = []
        
        for line in stdout_lines:
            if line.strip() == "PROFILER_RESULT_START":
                result_started = True
                continue
            elif line.strip() == "PROFILER_RESULT_END":
                break
            elif result_started:
                result_lines.append(line)
        
        if result_lines:
            return '\n'.join(result_lines)
        else:
            return f"Profiler Error: No result found in subprocess output\\nStdout: {result.stdout}\\nStderr: {result.stderr}"
            
    except subprocess.TimeoutExpired:
        return f"Profiler Error: Timed out after 600 seconds"
    except Exception as e:
        return f"Profiler Error: {str(e)}"
    finally:
        # Clean up temporary script
        try:
            os.unlink(script_path)
        except:
            pass
