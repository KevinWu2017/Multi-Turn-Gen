import os
import json
import pydra
from pydra import Config, REQUIRED

# 加载.env文件中的环境变量
# NOTE: Make sure this dotenv is loaded before any other imports of KernelBench or Caesar modules 
from dotenv import load_dotenv
load_dotenv()

from datasets import load_dataset

from logger import CaesarLogger
from states import WorkArgs
from state_machine import CaesarStateMachine
from orchestrator import GPUOrchestrator
from transitions_def import MyTransition
from utils import build_context_multi_turn

from src.dataset import construct_kernelbench_dataset
from src.prompt_constructor import prompt_generate_custom_cuda_from_prompt_template, prompt_generate_prompt_with_hardware_info_from_template
from src.utils import (
    extract_last_code,
    extract_first_code,
    extract_code_blocks,
    read_file,
)

from multi_turn_config import MultiTurnConfig

@pydra.main(base=MultiTurnConfig)
def main(config: MultiTurnConfig):
    config.set_preset_config(
        config.model_name if config.model_name is not None else "deepseek"
    )
    if config.dataset_src == "huggingface":
        dataset = load_dataset(config.dataset_name)
        curr_level_dataset = dataset[f"level_{config.level}"]
    elif config.dataset_src == "local":
        curr_level_dataset = construct_kernelbench_dataset(config.level)
    
    problem = curr_level_dataset[config.problem_id - 1]
    sample_id = 0
    orchestrator = GPUOrchestrator(num_gpus=1)
    work = WorkArgs(problem=problem, problem_id=str(config.problem_id), sample_id=sample_id)

    log_dir = os.path.join(config.log_dir_prefix, config.run_group, config.run_name, "problem_" + str(work.problem_id), "sample_" + str(work.sample_id))
    logger = CaesarLogger(log_dir, config, work, verbose=config.verbose, log_name=f"log.json")

    transitions = MyTransition()
    transitions._validate_transitions()

    caesar = CaesarStateMachine(
        transition_cfg=transitions,
        config=config,
        work=work,
        process_id=0,
        logger=logger,
        orchestrator=orchestrator,
    )

    ref_arch_src = read_file(problem)

    if config.use_gpu_info:
        caesar.initial_prompt = prompt_generate_prompt_with_hardware_info_from_template(ref_arch_src=ref_arch_src, gpu_name=caesar.config.gpu_name)
    else:
        caesar.initial_prompt = prompt_generate_custom_cuda_from_prompt_template(ref_arch_src=ref_arch_src)

    for turn_id in range(1, config.max_k + 1):
        if turn_id != 4:
            continue
        context = build_context_multi_turn(
            initial_prompt=caesar.initial_prompt,
            contexts=caesar.context,
            kernels=caesar.kernel_code,
            compiler_feedback=caesar.feedback,
            eval_result=caesar.eval_result,
            profiler_result=caesar.profiler_result,
            iteration=caesar.current_k,
            strategy=caesar.config.context_strategy,
            use_last_only=caesar.config.use_last_only,
            max_feedback_length=caesar.config.max_feedback_length,
        )
        print(f"=== Turn {turn_id} ===")
        print("[Context]")
        print(context)

if __name__ == "__main__":
    main()
