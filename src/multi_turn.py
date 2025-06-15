import os
import time

import pydra
from pydra import Config, REQUIRED
from typing import Dict, Optional

from datasets import load_dataset

from logger import CaesarLogger
from states import CaesarState, StateOutcome, Transition, WorkArgs
from state_machine import CaesarStateMachine
from orchestrator import GPUOrchestrator

from transitions_def import MyTransition, Reflection, Reflection_NVCC

from src.dataset import construct_kernelbench_dataset
from src.eval import eval_kernel_against_ref
from src.prompt_constructor import prompt_generate_custom_cuda_from_prompt_template
from src.utils import (
    extract_first_code, 
    query_server, 
    set_gpu_arch, 
    read_file, 
    create_inference_server_from_presets
)

from utils import check_result_exists, timeout

REPO_TOP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class MultiTurnConfig(Config):
    def __init__(self):

        self.dataset_src = REQUIRED
        self.dataset_name = "ScalingIntelligence/KernelBench"

        self.level = REQUIRED
        self.problem_id = REQUIRED

        # LLM configs
        self.server_type = "sglang"
        self.server_address = "10.0.16.46"
        self.server_port = 8001
        self.model_name = "Qwen/Qwen2.5-Coder-32B-Instruct"

        # decoding parameter
        self.greedy_sample = False
        self.temperature = 0.0
        self.top_p = 1.0  # set to consider all tokens
        self.top_k = 50  # set large for default
        self.num_completions = 1
        self.max_tokens = 4096

        # Eval Speciifc
        self.num_correct_trials = 5
        self.num_perf_trials = 100
        self.timeout = 600 # time out per round, set to 10 min 
        self.gpu_arch = ["Ampere"] 

        self.log_dir_prefix = os.path.join(REPO_TOP_DIR, "results/kernel_multi_turn")
        self.build_dir_prefix = os.path.join(REPO_TOP_DIR, "results/kernel_eval_build")

        self.run_group = "dummy_run_group"
        self.run_name = "dummy_run_name"

        self.verbose = True
        self.show_state = True
        self.measure_performance = True
        self.mock = False
        self.debug = False

        # multi-turn numbers
        self.max_k = 10
        self.context_strategy = ["reflection"]
        self.state_machine_strategy = "rerun" # default
        self.max_feedback_length = 100000 # in terms of characters, 10k, so much less in tokens
        # TODO: this is not implemented in the state machine yet
        self.use_last_only = False 

def start_single_caesar(
    work: WorkArgs,
    config: MultiTurnConfig,
    logger: CaesarLogger,
    process_id: Optional[int] = None,
    orchestrator: Optional[GPUOrchestrator] = None,
) -> int:
    """
    Start a single caesar process
    """
    
    print(f"Starting Caesar with process_id {process_id} and orchestrator {orchestrator}")
    # define state machine transition configs
    transitions = MyTransition()
    transitions._validate_transitions()

    caesar = CaesarStateMachine(
        transition_cfg=transitions,
        config=config,
        work=work,
        process_id=process_id,
        logger=logger,
        orchestrator=orchestrator,
    )
    returncode = caesar.run()
    return returncode

@pydra.main(base=MultiTurnConfig)
def main(
    config: MultiTurnConfig
):
    if config.dataset_src == "huggingface":
        dataset = load_dataset(config.dataset_name)
        curr_level_dataset = dataset[f"level_{config.level}"]
    elif config.dataset_src == "local":
        curr_level_dataset = construct_kernelbench_dataset(config.level)
    
    problem = curr_level_dataset[config.problem_id - 1]
    sample_id = 0
    orchestrator = GPUOrchestrator(num_gpus=1)
    print(f"Running problem {config.problem_id} {problem} with sample {sample_id}")
    work = WorkArgs(problem=problem, problem_id=str(config.problem_id), sample_id=sample_id)

    log_dir = os.path.join(config.log_dir_prefix, config.run_group, config.run_name, "problem_" + str(work.problem_id), "sample_" + str(work.sample_id))
    logger = CaesarLogger(log_dir, config, work, verbose=config.verbose, log_name=f"log.json")

    returncode = start_single_caesar(work=work, config=config, logger=logger, process_id=0, orchestrator=orchestrator)

if __name__ == "__main__":
    # main_orchestrator()
    main()