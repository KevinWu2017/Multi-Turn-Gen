import os
from typing import List
import json
from pydra import Config
import signal

from src.utils import read_file
from src.eval import KernelExecResult

###########################
# Prompt Construction


REPO_TOP_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
    )
)
KERNEL_BENCH_REPO_PATH = os.path.join(REPO_TOP_PATH, "KernelBenchInternal")

KERNEL_BENCH_PATH = os.path.join(REPO_TOP_PATH, "KernelBenchInternal", "KernelBench")


# These are from KernelBenchInternal/src/prompt_constructor.py
# overall problem state
PROBLEM_STATEMENT = """You write custom CUDA kernels to replace the pytorch operators in the given architecture to get speedups.\n\nYou have complete freedom to choose the set of operators you want to replace. You may make the decision to replace some operators with custom CUDA kernels and leave others unchanged. You may replace multiple operators with custom implementations, consider operator fusion opportunities (combining multiple operators into a single kernel, for example, combining matmul+relu), or algorithmic changes (such as online softmax). You are only limited by your imagination.\n
"""

PROBLEM_INSTRUCTION = """Optimize the architecture named Model with custom CUDA operators! Name your optimized output architecture ModelNew. Output the new code in codeblocks. Please generate real code, NOT pseudocode, make sure the code compiles and is fully functional. Just output the new model code, no other text, and NO testing code! \n
"""

REFLECTION_INSTRUCTION = """Given your previous generation(s), improve and optimize the architecture named Model with custom CUDA operators! Name your optimized output architecture ModelNew. Output the new code in codeblocks. Please generate real code, NOT pseudocode, make sure the code compiles and is fully functional. Just output the new model code, no other text, and NO testing code! \n
"""

REFLECTION_INSTRUCTION_LAST_ONLY = """Given your latest generation, improve and optimize the architecture named Model with custom CUDA operators! Name your optimized output architecture ModelNew. Output the new code in codeblocks. Please generate real code, NOT pseudocode, make sure the code compiles and is fully functional. Just output the new model code, no other text, and NO testing code! \n
"""


def prompt_generate_initial_from_template(ref_arch_src: str):
    example_ind = 1
    example_arch_path = os.path.join(
        KERNEL_BENCH_REPO_PATH, f"src/prompts/model_ex_{example_ind}.py"
    )
    example_new_arch_path = os.path.join(
        KERNEL_BENCH_REPO_PATH, f"src/prompts/model_new_ex_{example_ind}.py"
    )
    example_arch = read_file(example_arch_path)
    example_new_arch = read_file(example_new_arch_path)

    return prompt_generate_initial(ref_arch_src, example_arch, example_new_arch)


def prompt_generate_initial(
    arch_src: str, example_arch_src: str, example_new_arch_src: str
) -> str:
    prompt = PROBLEM_STATEMENT

    if example_arch_src != "" and example_new_arch_src != "":
        prompt += f"""
        Here's an example to show you the syntax of inline embedding custom CUDA operators in torch: The example given architecture is: \n
        ``` \n
        {example_arch_src}
        ``` \n
        The example new arch with custom CUDA kernels looks like this: 
        ```
        {example_new_arch_src}
        ``` \n
        """

    prompt += f"""
    You are given the following architecture: \n
    ```
    {arch_src}
    ```
    """
    # prompt += PROBLEM_INSTRUCTION
    return prompt


def build_context_multi_turn(
    initial_prompt: str,
    contexts: dict,
    kernels: dict,
    compiler_feedback: dict,
    eval_result: dict,
    profiler_result: dict,
    iteration: int,
    strategy: List[str] = [],
    use_last_only: bool = False,
    max_feedback_length: int = 4000,
) -> str:
    """
    Build the current context for the given iteration and strategy
    """
    prompt = initial_prompt

    # Add feedback
    if "eval_result" in strategy:
        if iteration == 1:
            prompt += "\n\n"
            prompt += PROBLEM_INSTRUCTION
        else:
            prompt += f"Here is your latest generation:\n"
            prompt += f"```\n{kernels[iteration - 1]}\n```\n"
            prompt += construct_programatic_prompt_feedback(
                compiler_feedback=compiler_feedback[iteration - 1],
                exec_result=eval_result[iteration - 1],
                profiler_feedback=(
                    profiler_result[iteration - 1]
                    if iteration - 1 in profiler_result
                    else ""
                ),
                max_feedback_length=max_feedback_length,
                use_pytorch_profiler=("profiler" in strategy),
            )
        return prompt

    # Reflection
    # Iterations are 1-indexed.
    for i in range(1, iteration, 1):

        # Add what you generated
        if "reflection" in strategy:

            # Only use the latest generation
            if use_last_only:
                if i < iteration - 1:
                    continue
                else:
                    prompt += f"Here is your latest generation:\n"
                    prompt += f"```\n{kernels[i]}\n```\n"
            else:
                # Include all previous generations

                # Special case for the first generation
                if i == 1:
                    prompt += f"Here are your previous generations:\n"

                # add kernels
                if i in kernels:
                    prompt += f"Generation {i}:\n```\n{kernels[i]}\n```\n\n"

        # Human feedback
        if "human_feedback" in strategy:
            print(f"\nPlease provide feedback for kernel {i}:")
            user_feedback = input().strip()
            if user_feedback:
                prompt += f"This is expert human feedback on your previously generated kernel #{i}:\n"
                prompt += f"{user_feedback}\n\n"

    # Instruction prompt
    if iteration == 1:  # for intial
        prompt += PROBLEM_INSTRUCTION
    else:
        if "reflection" in strategy:
            if use_last_only:
                prompt += REFLECTION_INSTRUCTION_LAST_ONLY
            else:
                prompt += REFLECTION_INSTRUCTION
        else:
            raise NotImplementedError("Anything else is not implemented yet")
            prompt += PROBLEM_INSTRUCTION

    return prompt


EVAL_RESULT_INSTRUCTION_LAST_ONLY = """Name your new improved output architecture ModelNew. Output the new code in codeblocks. Please generate real code, NOT pseudocode, make sure the code compiles and is fully functional. Just output the new model code, no other text, and NO testing code! \n
"""


def construct_programatic_prompt_feedback(
    compiler_feedback: str,
    exec_result: KernelExecResult,
    profiler_feedback: str = "",
    max_feedback_length: int = 2000,
    use_pytorch_profiler: bool = False,
) -> str:
    """
    Construct programatic feedback from the given exec_result
    input:
    - compiler_feedback: str, feedback from the compiler
    - exec_result: KernelExecResult, the result of the execution
    - profiler_feedback: str, table breakdown from the PyTorch profiler; only used if use_pytorch_profiler is True
    - max_log_length: int, the maximum length of the log to put back into context
    - use_pytorch_profiler: bool, whether to use the PyTorch profiler
    """
    feedback = "Your generated architecture ModelNew and kernel was evaluated on GPU and checked against the reference architecture Model. \n\n"
    feedback += "Here is the Evaluation Result: \n"

    # we do not want hardware information to influence this
    metadata_without_hw_info = exec_result.metadata
    metadata_without_hw_info.pop("hardware", None)
    metadata_without_hw_info.pop("device", None)

    # Case 1: Compilation Failure
    if not exec_result.compiled:
        feedback += "Your kernel failed to compile.\n\n"
        # TODO: restrict length of compiler feedback
        feedback += f"Here is the compiler logs\n\n:"
        feedback += f"{compiler_feedback[:max_feedback_length]}\n\n"
        for key, value in metadata_without_hw_info.items():
            feedback += f"{key}: {value}\n"
        feedback += EVAL_RESULT_INSTRUCTION_LAST_ONLY
        feedback += "Please fix the errors and try again.   "
        return feedback

    # Special Case: CUDA Error
    if "cuda_error" in metadata_without_hw_info:
        feedback += "Your kernel failed to run. \n\n"
        feedback += f"Here is the CUDA error: {exec_result.metadata['cuda_error']} \n\n"
        feedback += EVAL_RESULT_INSTRUCTION_LAST_ONLY

        feedback += "Please fix the errors and try again."
        return feedback

    # Special Case: Time-out
    if "timeout_error" in metadata_without_hw_info:
        feedback += "Your kernel execution timed out. \n\n"
        feedback += EVAL_RESULT_INSTRUCTION_LAST_ONLY
        feedback += "Please fix the errors and try again."

        return feedback

    # Case 2: Compiled, But not Correct
    if not exec_result.correctness:
        feedback += "Your kernel failed to produce the correct output, compared to the reference architecture.\n\n"
        feedback += f"Here is the correctness feedback: \n\n"
        # add metadata objects
        for key, value in metadata_without_hw_info.items():
            feedback += f"{key}: {value}\n"
        feedback += EVAL_RESULT_INSTRUCTION_LAST_ONLY
        feedback += (
            "Please try generating ModelNew again, while fixing the correctness issues."
        )
        return feedback

    # Case 3: Runtime Success
    if exec_result.correctness:
        feedback += (
            "Your kernel executed successfully and produced the correct output. \n\n"
        )
        feedback += (
            f"Here is your wall clock time: {exec_result.runtime} milliseconds \n\n"
        )

        # ADD PROFILER
        if use_pytorch_profiler:
            assert len(profiler_feedback) > 0, "Profiler feedback is empty"
            feedback += f"Your Kernel was profiled with the PyTorch Profiler over many iterations, below is a table breakdown of the CUDA kernel execution time: \n\n"

            feedback += f"```\n{profiler_feedback[:max_feedback_length]}\n```\n\n"

        feedback += EVAL_RESULT_INSTRUCTION_LAST_ONLY
        feedback += "Please rewrite the entire kernel to be as fast as possible. \n\n"
        return feedback

    raise ValueError("[Programatic Feedback] You should not reach here")
    return None


def ensure_json_serializable(obj):
    """
    Recursively convert any object into a JSON serializable format.
    Handles nested dictionaries, lists, and custom objects.

    Args:
        obj: Any Python object
    Returns:
        JSON serializable version of the object
    """
    if isinstance(obj, dict):
        return {k: ensure_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [ensure_json_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(ensure_json_serializable(item) for item in obj)
    elif isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    elif hasattr(obj, "__dict__"):  # Handle custom objects
        return ensure_json_serializable(obj.__dict__)
    else:
        return str(obj)  # Convert anything else to string


def check_result_exists(
    log_dir_prefix: str, run_group: str, run_name: str, problem_id: int, sample_id: int
) -> bool:
    """
    Check if the result for the given run_name, sample_id and problem_id exists.
    """
    path = os.path.join(
        log_dir_prefix,
        run_group,
        run_name,
        f"problem_{str(problem_id)}",
        f"sample_{str(sample_id)}",
        "DONE",
    )
    # print(f"Checking if {path}: {os.path.exists(path)}") # DEBUG

    return os.path.exists(path)


def check_result_exists_run_path(
    run_path: str, problem_id: int, sample_id: int
) -> bool:
    """
    Check if the result for the given run_name, sample_id and problem_id exists.
    """
    path = os.path.join(
        run_path, f"problem_{str(problem_id)}", f"sample_{str(sample_id)}", "DONE"
    )
    # print(f"Checking if {path}: {os.path.exists(path)}") # DEBUG

    return os.path.exists(path)


def get_run_group_stats(log_dir_prefix: str, run_group: str) -> dict:
    """
    Get the stats for the given run_group, check how many runs have finished
    """
    run_group_path = os.path.join(log_dir_prefix, run_group)
    run_group_stats = {}

    for run_name in os.listdir(run_group_path):
        # go through each run group

        num_evaluated = 0
        run_path = os.path.join(run_group_path, run_name)

        for problem_id in os.listdir(run_path):
            problem_path = os.path.join(run_path, problem_id)
            for sample_id in os.listdir(problem_path):
                sample_path = os.path.join(problem_path, sample_id)
                if os.path.exists(os.path.join(sample_path, "DONE")):
                    num_evaluated += 1

        run_group_stats[run_name] = num_evaluated

    return run_group_stats


class timeout:
    def __init__(self, seconds):
        self.seconds = seconds

    def handle_timeout(self, signum, frame):
        raise TimeoutError(f"Operation timed out after {self.seconds} seconds")

    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(int(self.seconds))

    def __exit__(self, type, value, traceback):
        signal.alarm(0)
