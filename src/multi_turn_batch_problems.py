import os
import time
import multiprocessing as mp

import pydra
from pydra import Config, REQUIRED
from typing import Dict, Optional
from tqdm import tqdm
import traceback
import queue

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

def create_work_queue(config: MultiTurnConfig):
    works :list[WorkArgs] = []
    # Decide which problems we should run 
    
    if config.dataset_src == "huggingface":
        dataset = load_dataset(config.dataset_name)
        curr_level_dataset = dataset[f"level_{config.level}"]
    elif config.dataset_src == "local":
        curr_level_dataset = construct_kernelbench_dataset(config.level)

    num_problems = len(curr_level_dataset)

    for problem_id in range(1, num_problems + 1):
        for sample_id in range(config.num_samples):
            if config.dataset_src == "huggingface":
                curr_problem_row = curr_level_dataset.filter(lambda x: x["problem_id"] == config.problem_id)
                ref_arch_src = curr_problem_row["code"][0]
                problem_name = curr_problem_row["name"][0]

            elif config.dataset_src == "local":
                problem_idx_in_dataset = config.problem_id - 1 # due to dataset list being 0-indexed locally
                ref_arch_path = curr_level_dataset[problem_idx_in_dataset]
                problem_name = os.path.basename(ref_arch_path)
                ref_arch_src = read_file(ref_arch_path)

            if check_result_exists(config.log_dir_prefix, config.run_group, config.run_name, problem_id, sample_id):
                print(f"[SKIP] Result for run {config.run_name} problem {problem_id} sample {sample_id} already exists... skipping")
                continue
            else:
                curr_work = WorkArgs(problem=ref_arch_src, problem_id=str(problem_id), sample_id=sample_id)
                works.append(curr_work)
            
    print(f"Created {len(works)} works")
    return works

def run_caesar_with_timeout(work, config, process_id, orchestrator, logger, result_queue):
    """Helper function to run caesar in a separate process"""
    try:
        returncode = start_single_caesar(work, config, logger=logger, process_id=process_id, orchestrator=orchestrator)
        result_queue.put(returncode)
    except Exception as e:
        print(f"[Error] Inner process encountered error: {e}")
        traceback.print_exc()
        result_queue.put(1)  # Error code

def worker_process(
        mp_queue: mp.Queue,
        config: MultiTurnConfig,
        progress_queue: mp.Queue,
        process_id: Optional[int] = None,
        orchestrator: Optional[GPUOrchestrator] = None,
    ):
    """
    This is process to be multi-processed
    This gets called in the mp loop as a process. It runs a job
    then grabs a new one from the queue of jobs (atomically).
    """
    while True:
        print(f"[Heartbeat] Process {process_id} start of loop")
        try:
            work = mp_queue.get(timeout=1)
            # if queue is empty it shuts down    
            if work is None: 
                print(f"[Shutdown] Worker {process_id} received shutdown signal")
                break
        
            try:
                print(f"[Launch] Worker {process_id} launching work {work}")

                # Initialize logger here.
                log_dir = os.path.join(config.log_dir_prefix, config.run_group, config.run_name, "problem_" + str(work.problem_id), "sample_" + str(work.sample_id))
                logger = CaesarLogger(log_dir, config, work, verbose=config.verbose, log_name=f"log.json")
                
                # Create a queue for the inner process result
                result_queue = mp.Queue()
                
                # Create and start the inner process
                inner_process = mp.Process(
                    target=run_caesar_with_timeout,
                    args=(work, config, process_id, orchestrator, logger, result_queue)
                )
                inner_process.start()

                try:
                    with timeout(config.timeout * config.max_k):  # 10 minutes per round before timeout
                        inner_process.join()  # Wait for process to complete
                        if inner_process.is_alive():
                            inner_process.terminate()
                            inner_process.join(timeout=1)  # Wait briefly for termination
                            if inner_process.is_alive():
                                inner_process.kill()  # Force kill if still alive
                            raise TimeoutError("Process timed out")
                        
                        # Get the result if process completed
                        returncode = result_queue.get(timeout=1)

                        if returncode == 0:
                            progress_queue.put(1)
                            print(f"[Finish] Worker {process_id} finished work {work}")
                        else:
                            print(f"[Orchestrator] Worker {process_id} encountered error: {returncode}, adding Work {work} back to queue")
                            # Make sure inner process is terminated
                            if inner_process.is_alive():
                                inner_process.terminate()
                                inner_process.join(timeout=1)
                                if inner_process.is_alive():
                                    inner_process.kill()
                            mp_queue.put(work)
                            print(f"Work {work} added back to mp_queue.")
                            
                            continue  # Continue to next work item instead of exiting


                except TimeoutError as e:
                    inner_process.terminate()

                    # TODO: Write to the log.
                    logger._save_timeout_eval("Time limit reached: Kernel either hung or timed out.")

                    print(f"[Orchestrator] Worker {process_id} timed out, shutting down... but adding Work {work} back to queue")  
                    mp_queue.put(work)

                    raise TimeoutError(f"Process timed out: {e}")

            except TimeoutError as e:
                print(f"[Timeout] Worker {process_id} GPU acquisition timed out: {e}")

                # Write a timeout
                log_dir = os.path.join(config.log_dir_prefix, config.run_group, config.run_name, "problem_" + str(work.problem_id), "sample_" + str(work.sample_id))

                os.makedirs(log_dir, exist_ok=True)
                with open(os.path.join(log_dir, "TIMEOUT"), "w") as f:
                    pass
                # Optionally requeue the work item
                # mp_queue.put(work)
                continue
                
            except Exception as e:
                print(f"[Error] Worker {process_id} encountered error: {e}")
                traceback.print_exc()
                # Optionally log or handle specific errors differently
                continue
        except queue.Empty:
            # if queue is empty it shuts down
            time.sleep(10)
            continue
            # break

@pydra.main(base=MultiTurnConfig)
def main_orchestrator(config: MultiTurnConfig):
    mp.set_start_method('spawn', force=True)

    orchestrator = GPUOrchestrator(num_gpus=config.num_gpus)

    processes = []
    job_queue = mp.Queue()
    progress_queue = mp.Queue()  # New queue for progress tracking
    total_work = create_work_queue(config)
    total_work_count = len(total_work)

    if total_work_count == 0:
        print("[Orchestrator] No work to do, shutting down and exiting...")
        return
    
    for work in total_work:
        job_queue.put(work)

    try:
        # Start worker processes
        for worker_id in range(config.num_workers):
            p = mp.Process(
                target=worker_process, 
                args=(job_queue, config, progress_queue),
                kwargs={"process_id": worker_id, "orchestrator": orchestrator}
            )
            p.start()
            processes.append(p)

        # Progress bar for overall progress
        with tqdm(total=total_work_count, desc="Overall Progress") as pbar:
            completed_work = 0
            while completed_work < total_work_count:
                progress_queue.get()  # Wait for a work item to complete
                completed_work += 1
                pbar.update(1)

        # Wait for all processes to complete
        for p in processes:
            p.terminate()
            p.join()

    except KeyboardInterrupt:
        print("\n[Orchestrator] Shutting down...")
        # Terminate all processes
        for p in processes:
            p.terminate()
            p.join()

if __name__ == "__main__":
    main_orchestrator()