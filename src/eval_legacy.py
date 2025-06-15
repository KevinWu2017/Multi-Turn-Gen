import torch
from pydra import Config
import os
import re

from torch.profiler import profile, record_function, ProfilerActivity
import sys
# Add KernelBench to path for imports
KERNELBENCH_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "KernelBench")
if KERNELBENCH_PATH not in sys.path:
    sys.path.insert(0, KERNELBENCH_PATH)
from src import eval as kernel_eval
from src import utils as kernel_utils
from states import CaesarState, StateOutcome, WorkArgs
from utils import timeout

"""
Additional Eval Code
"""


def get_kernel_hash(kernel_src: str) -> str:
    return str(hash(kernel_src))

def compile_single_sample(kernel_src: str, config: Config, build_dir: str, timeout_seconds: int = 480):
    """
    CPU Pre compile kernel and capture any errors
    """    
    kernel_utils.set_gpu_arch(config.gpu_arch)

    # Withih this build dir, build_dir/run_name/problem_id/sample_id/kernel_hash/...
    # we have bin, .c,.cu, .so,.py (temp just for this kernel)
    kernel_hash = get_kernel_hash(kernel_src)
    kernel_build_dir = os.path.join(build_dir, kernel_hash)
        
    try:
        with timeout(timeout_seconds):
            returncode, stdout, err = kernel_eval.build_compile_cache_with_capturing(
                custom_model_src=kernel_src,
                verbose=config.verbose,
                build_dir=kernel_build_dir,
            )
                
            return returncode, stdout, err
    except TimeoutError:
        print(f"[WARNING] Compilation timed out after {timeout_seconds} seconds")
        return -1, f"Compilation timed out after {timeout_seconds} seconds", f"Compilation timed out after {timeout_seconds} seconds"
    except Exception as e:
        print(f"[WARNING] Last level catch when CPU pre-compiling kernel: Some issue while compiling and attempting to cache for kernel: {e} ")
        return -1, str(e), str(e)


def evaluate_single_sample_src(ref_arch_src: str, kernel_src: str, configs: Config, build_dir: str, device: torch.device, timeout_seconds: int = 480) -> kernel_eval.KernelExecResult:
    """
    Evaluate a single sample source code against a reference architecture source code.
    Args:
        timeout_seconds: Maximum time in seconds to wait for evaluation (default 8 minutes)
    """
    # TODO: Figure out how to compile correctly. Recompile for now
    kernel_hash = get_kernel_hash(kernel_src)
    build_dir = os.path.join(build_dir, kernel_hash)
 
    # temp_name = kernel_hash[1:]
    # kernel_src = re.sub(r'name="[^"]*"', f'name="{temp_name}"', kernel_src)

    try:
        with timeout(timeout_seconds):
            eval_result = kernel_eval.eval_kernel_against_ref(
                original_model_src=ref_arch_src,
                custom_model_src=kernel_src,
                measure_performance=configs.measure_performance,
                verbose=configs.verbose,
                num_correct_trials=configs.num_correct_trials,
                num_perf_trials=configs.num_perf_trials,
                # move this to config in monkeys
                build_dir=build_dir,
                device=device
            )
            return eval_result
    except TimeoutError:
        print(f"[WARNING] Evaluation timed out after {timeout_seconds} seconds")
        metadata = {"timeout_error": f"Evaluation timed out after {timeout_seconds} seconds",
                    "hardware": torch.cuda.get_device_name(device=device),
                    "device": str(device)
                    }
        metadata = kernel_eval.check_metadata_serializable(metadata)
        eval_result = kernel_eval.KernelExecResult(compiled=False, correctness=False,
                                            metadata=metadata)
        return eval_result
    except Exception as e:
        print(f"[WARNING] Last level catch: Some issue evaluating for kernel: {e} ")
        if "CUDA error" in str(e): 
            # NOTE: count this as compilation failure as it is not runnable code
            metadata = {"cuda_error": f"CUDA Error: {str(e)}",
                        "hardware": torch.cuda.get_device_name(device=device),
                        "device": str(device)
                        } # for debugging
            
            metadata = kernel_eval.check_metadata_serializable(metadata)
            eval_result = kernel_eval.KernelExecResult(compiled=False, correctness=False, 
                                                metadata=metadata)
            return eval_result
        else:
            metadata = {"other_error": f"error: {str(e)}",
                        "hardware": torch.cuda.get_device_name(device=device),
                        "device": str(device)
                        } # for debugging
            metadata = kernel_eval.check_metadata_serializable(metadata) 
            eval_result = kernel_eval.KernelExecResult(compiled=False, correctness=False, 
                                                metadata=metadata)
            return eval_result
        


def get_torch_profiler_info(ref_arch_src: str, 
                            kernel_src: str, 
                            build_dir: str, 
                            device: torch.device, 
                            num_trials: int = 100,
                            table_row_limit: int = 10,
                            seed_num: int = 42)->str:
    """
    Get the profiler info for a particular kernel
    Given a KernelBench solution to a problem, we want to profile the kernel

    ref_arch_src: str, the source code of the reference architecture; we use this to get the inputs
    kernel_src: str, the source code of the kernel; this will be compiled and used to get the model
    build_dir: str, the directory to build the custom kernel
    device: torch.device, the device to run the profiler on
    num_trials: int, the number of trials to run for Torch profiling
    table_row_limit: int, the number of rows to display in the profiler table
    seed_num: int to initiliaze on device random seed


    Notes about profiling:
        - We do not set p.toggle_collection_dynamic explicitly, 
        - We only collect CUDA activity (ProfilerActivity.CUDA), as we are only interested in the kernel
        
    """
    assert torch.cuda.is_available(), "CUDA is not available, cannot run Torch Profiler"

    kernel_hash = get_kernel_hash(kernel_src)
    build_dir = os.path.join(build_dir, kernel_hash)

    context = {}
    _, get_init_inputs, get_inputs = kernel_eval.load_original_model_and_inputs(
        ref_arch_src, context
    )

    kernel_eval.set_seed(seed_num)
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
            active=num_trials,
        )

        with profile(
            activities=[ProfilerActivity.CUDA],
            schedule=profiling_scheduler,
        ) as prof:
            for _ in range(num_trials):
            
                output = model(*inputs)
                prof.step()

        profiler_output = prof.key_averages().table(sort_by='cuda_time_total', 
                                                    row_limit=table_row_limit)
        
    return profiler_output