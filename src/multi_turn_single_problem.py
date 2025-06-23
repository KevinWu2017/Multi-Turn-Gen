import os
import time
import multiprocessing as mp

import pydra
from pydra import Config, REQUIRED
from typing import Dict, Optional

from datasets import load_dataset

from logger import CaesarLogger
from states import WorkArgs
from state_machine import CaesarStateMachine
from orchestrator import GPUOrchestrator

from transitions_def import MyTransition, Reflection, Reflection_NVCC

from src.dataset import construct_kernelbench_dataset

from multi_turn_config import MultiTurnConfig

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
    config.set_preset_config(config.model_name if config.model_name is not None else "deepseek")

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

    return returncode

if __name__ == "__main__":
    main()