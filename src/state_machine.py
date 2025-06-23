"""
State Machine


"""
import json
import time
import random
from typing import Optional
from pydra import Config
from enum import Enum
import copy
import os
import sys
import torch
import signal

from states import CaesarState, StateOutcome, WorkArgs
from logger import CaesarLogger
from utils import build_context_multi_turn, prompt_generate_initial_from_template, timeout
from orchestrator import GPUOrchestrator
# Here temporaryly use caesar's legacy eval code, KernelBench's new eval code uses torch compile to compile automatically 
from eval_legacy import (
    compile_single_sample, 
    evaluate_single_sample_src, 
    get_kernel_hash, 
    get_torch_profiler_info
)

from src.eval import build_compile_cache_with_capturing
from src.prompt_constructor import prompt_generate_custom_cuda_from_prompt_template, prompt_generate_prompt_with_hardware_info_from_template
from src import eval as kernel_eval
from src.utils import (
    extract_last_code,
    extract_first_code,
    extract_code_blocks,
    read_file,
)
from src.inference_helper import create_inference_client, do_inference


def show_current_state(round: int, state: CaesarState, show_state: bool):
    if show_state:
        print(f"[StateMachine] I am in round {round} state: {state}")


class CaesarStateMachine:
    def __init__(
        self,
        transition_cfg: dict,
        config: Config,
        work: WorkArgs,
        logger: CaesarLogger,
        process_id: Optional[int] = None,
        orchestrator: Optional[GPUOrchestrator] = None,
        max_gpu_retries: int = 3,
    ):
        self.state = CaesarState.START_STATE
        self.outcome = StateOutcome.Start

        self.current_k = 0
        self.max_k = config.max_k  # TODO: is an a arg
        self.transition_cfg = transition_cfg
        self.config = config

        # KernelBench
        self.run_name = config.run_name
        self.run_group = config.run_group
        self.problem_id = work.problem_id
        self.sample_id = work.sample_id
        self.problem = work.problem
        self.state_machine_strategy = config.state_machine_strategy

        self.ref_arch_src = ""  # problem in plain text

        # LLM state information
        self.curr_context: str = ""

        self.context: dict[int, str] = {}
        self.model_response: dict[int, str] = {}
        self.kernel_code: dict[int, str] = {}
        self.eval_result: dict[int, str] = {}
        self.profiler_result: dict[int, str] = {}

        # external feedback
        self.feedback: dict[int, str] = {}

        # Timeout stuff
        self.correct_state_timeout = config.timeout
        self.compile_state_timeout = config.timeout
        self.max_gpu_retries = max_gpu_retries
        self.gpu_retry_count = 0

        # Add logger initialization

        # this is problem / sample specific
        self.log_dir = os.path.join(config.log_dir_prefix, config.run_group, config.run_name, "problem_" + str(work.problem_id), "sample_" + str(work.sample_id))
        self.logger = logger # CaesarLogger(self.log_dir, config, work, verbose=config.verbose, log_name=f"log.json")

        # Check if run is already finished
        if os.path.exists(os.path.join(self.log_dir, "DONE")):
            print(f"[SKIP] Run {self.run_name} {self.problem_id} {self.sample_id} already finished... skipping")
            return
        # Check if previous run exists
        elif os.path.exists(os.path.join(self.log_dir, "log.json")):
            print(f"[RECOVER] Run was not finished, loading existing partial results from {self.log_dir}")
            self.load_from_previous_run()

        # Set up orchestrator and MP
        self.process_id = process_id
        self.orchestrator = orchestrator

        self.show_state = config.show_state
        self.inference_client = None  # This will be set in pre_run

        # Set up KernelBench as context, problem prompts etc.
        self.pre_run()

    def load_from_previous_run(self):
        """
        Load previous information from a prior run if it exists in self.log_dir.
        """
        log_file = os.path.join(self.log_dir, "log.json")
        with open(log_file, 'r') as f:
            saved_log = json.load(f)
        print(saved_log.keys())

        # Load turn data
        for turn in range(1, self.max_k+1, 1): 
            turn_str = str(turn)

            # The first turn that is not recorded in the log
            # Assumption: if it is saved in the file, it is finished
            self.current_k = turn
            if turn_str not in saved_log:
                self.current_k = turn - 1
                break

            # current turn
            turn_data = saved_log[turn_str]
            
            self.context[turn] = turn_data.get("context", "")
            self.model_response[turn] = turn_data.get("model_response", "")
            self.kernel_code[turn] = turn_data.get("kernel_code", "")
            self.feedback[turn] = turn_data.get("feedback", "")
            self.eval_result[turn] = turn_data.get("eval_result", "")
            self.profiler_result[turn] = turn_data.get("profiler_result", "")

            # If these are empty, this run was corrupted somehow.
            if self.context[turn] == "" or self.model_response[turn] == "" or (self.kernel_code[turn] == "" and self.feedback_code[turn] == ""):
                self.current_k = turn - 1
                break
            
            # In an actually valid run, convert str to EvalResult
            self.eval_result[turn] = self.logger.exec_log_to_obj(self.eval_result[turn])

            self.logger.log_turn(
                turn=turn,
                context=self.context[turn],
                model_response=self.model_response[turn],
                kernel_code=self.kernel_code[turn],
                feedback=self.feedback[turn],
                eval_result=self.eval_result[turn],
                profiler_result=self.profiler_result[turn],
                last=False
            )

            if self.config.verbose:
                print(f"[RECOVER] loaded in data from iteration {turn}")

        print(f"[RECOVER] Resuming from round {self.current_k}")


    def pre_run(self):
        """
        Set up code before starting the state machine.
        """
        if not os.path.exists(self.config.log_dir_prefix):
            os.makedirs(self.config.log_dir_prefix, exist_ok=True)

        # assume each problem is a path
        # problem_path_prefix = "../"  # to KernelBench directory
        # problem_path = os.path.join(problem_path_prefix, self.problem)
        self.ref_arch_src = read_file(self.problem)

        # self.initial_prompt = prompt_generate_initial_from_template(self.ref_arch_src)
        self.initial_prompt = prompt_generate_custom_cuda_from_prompt_template(ref_arch_src=self.ref_arch_src)

        self.build_dir = os.path.join(
            self.config.build_dir_prefix,
            self.run_group,
            self.run_name,
            "problem_" + str(self.problem_id),
            "sample_" + str(self.sample_id),
        )

        self.inference_client = create_inference_client(
            server_type= self.config.server_type,
            server_address=self.config.server_address,
            server_port=self.config.server_port,
            model_name=self.config.model_name,
            is_reasoning_model=self.config.is_reasoning_model,
        )
        

    def run(self) -> int:
        """
        Main state machine event loop.
        """

        # Check if run is already finished
        if os.path.exists(os.path.join(self.log_dir, "DONE")):
            print(f"[SKIP] Run {self.run_name} {self.problem_id} {self.sample_id} already finished... skipping")
            return

        while self.current_k <= self.max_k:
            match self.state:
                case CaesarState.START_STATE:
                    self.start_logic()
                case CaesarState.GENERATE_STATE:
                    self.generate_logic()
                case CaesarState.COMPILE_STATE:
                    self.compile_logic()
                case CaesarState.CORRECT_STATE:
                    self.correct_logic()
                case CaesarState.PERFORMANCE_STATE:
                    self.performance_logic()
                case CaesarState.FINISH_STATE:
                    """Special case where we finish early. Usually for errors."""
                    self.finish_logic()

                    # It returned with a failure.
                    return 1

                case _:
                    raise ValueError(f"Invalid state to be in: {self.state}")

            self.state = self.transition()

        self.state = CaesarState.FINISH_STATE
        self.finish_logic()

        # Edge case with empty dict. Hacky, but works.
        self.logger.clean_log()

        # It returned correctly.
        return 0

    def transition(self):
        """
        Based on transition config, move from current
        state to next state in StateMachine.
        returns the next state to be in
        """

        # we are at a curr state and with a partiuclar outcome
        # now we go transition (curr_state, outcome) -> next_state
        # outcome_to_state
        outcome_to_state: dict[StateOutcome, CaesarState] = self.transition_cfg
        return outcome_to_state[self.outcome]

        # state , possible outcomes
        # (state, particular outcome) [logic] -> next state

        # What are the transitions from outcome to next state
        # 1:1 mapping

    # Logic for each state
    def start_logic(self):
        """
        Logic for the start state.
        Increment the round number.
        """
        # If we're moving to a new turn, log the previous turn's data
        if self.current_k > 0 and self.current_k <= self.max_k:
            # this is logging the previous round's data
            self.logger.log_turn(
                self.current_k,
                self.context.get(self.current_k, ""),
                self.model_response.get(self.current_k, ""),
                self.kernel_code.get(self.current_k, ""),
                self.feedback.get(self.current_k, ""),
                self.eval_result.get(self.current_k, ""),
                self.profiler_result.get(self.current_k, ""),
            )

            # copy the context from the previous round
            # self.context[self.current_k] = copy.deepcopy(self.context[self.current_k - 1])

            # problem , g1, g2, gi-1
            # Update current context even more.

        # Increase the round number
        self.current_k += 1
        self.logger.update_turn(self.current_k)

        # Update current context
        self.curr_context = build_context_multi_turn(
            initial_prompt=self.initial_prompt,
            contexts=self.context,
            kernels=self.kernel_code,
            compiler_feedback=self.feedback,
            eval_result=self.eval_result,
            profiler_result=self.profiler_result,
            iteration=self.current_k,
            strategy=self.config.context_strategy,
            use_last_only=self.config.use_last_only,
            max_feedback_length=self.config.max_feedback_length,
        )
        self.context[self.current_k] = self.curr_context

        show_current_state(self.current_k, self.state, self.config.show_state)

        self.outcome = StateOutcome.Start
        # import pdb; pdb.set_trace()

    def generate_logic(self):
        """
        Logic for the generate state.
        Query LLM given context and generate kernel.
        """
        show_current_state(self.current_k, self.state, self.config.show_state)

        if self.config.mock:
            model_response = "Some dummy response."
            self.model_response[self.current_k] = model_response
            self.kernel_code[self.current_k] = "print('EXAMPLE KERNEL')"

        else:  # actual querying API
            # model_response = self.inference_server(self.curr_context)
            print("Current Context[len=", len(self.curr_context), "]: ", self.curr_context)
            self.logger.log_on_turn("context", self.curr_context)
            model_response = do_inference(
                client=self.inference_client,
                prompt=self.curr_context,
                model_name=self.config.model_name,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                top_k=self.config.top_k,
                max_tokens=self.config.max_tokens,
                is_reasoning_model=self.config.is_reasoning_model,
                stream=True
            )
            self.logger.log_on_turn("model_response", model_response)
            self.model_response[self.current_k] = model_response
            kernel_code = extract_last_code(
                model_response, ["python", "cpp"]
            )

            assert kernel_code is not None and len(kernel_code) > 0, "[Eroror] Kernel code is Empty, model generation FAILED"
            # NOTE: should we discard this then?

            self.kernel_code[self.current_k] = kernel_code

            # For handling timeouts, now save
            self.logger.log_on_turn("kernel_code", self.kernel_code[self.current_k])

        # Update global state info
        self.outcome = StateOutcome.Generate
        # TODO: WE WILL ACTUALLY EXTRACT

    def compile_logic(self, finish: bool = False):
        """
        Logic for the compile state.
        """
        # CPU Precompile
        # it corrects / fails
        # correct -> keep going
        # fail -> update context, feedback

        show_current_state(self.current_k, self.state, self.config.show_state)

        returncode, stdout, err = compile_single_sample(
            kernel_src=self.kernel_code[self.current_k],
            config=self.config,
            build_dir=self.build_dir,
            timeout_seconds=self.compile_state_timeout
        )

        if self.config.verbose:
            print("Compile returncode", returncode)
            print("Compile stdout", stdout)
            print("Compile stderr", err)
        
        compiler_feedback = None
        if returncode == 0:
            self.outcome = StateOutcome.CPUCompileSuccess
            self.feedback[self.current_k] = ""
        else:
            self.outcome = StateOutcome.CPUCompileFail
            compiler_feedback = f"Compilation failed:\nstdout: {stdout}\nstderr: {err}"
            self.feedback[self.current_k] = compiler_feedback

            self.eval_result[self.current_k] = kernel_eval.KernelExecResult(
                compiled=False,
                correctness=False,
                metadata={"compiler_error": compiler_feedback,
                          "hardware": "cpu",
                          "device": "cpu"
                        }
                )

        if self.config.mock:
            compiler_feedback = f"Mock compilation happened. Returning {self.outcome}"
            self.feedback[self.current_k] = compiler_feedback

        if finish:
            self.outcome = StateOutcome.Finish
        else:
            # For handling timeouts, now save
            self.logger.log_on_turn("feedback", None if not compiler_feedback else compiler_feedback)

    def correct_logic(self, finish: bool = False):
        """
        Correct the kernel code.
        Args: finish - if this is the last state correctness test
        """
        show_current_state(self.current_k, self.state, self.config.show_state)

        if self.orchestrator is not None:
            print(f"[Worker] Process {self.process_id} requesting GPU...")
            
            with self.orchestrator.reserve_gpu() as gpu_id:
                try:
                    print(f"[Worker] Process {self.process_id} acquired GPU {gpu_id}")
                    # Put Eval Code Here
                    device = torch.device(f"cuda:{gpu_id}")
                
                    if self.config.mock:
                        # Simulate some GPU computation
                        work_time = random.uniform(1, 5)
                        torch.randn(100).to(device)
                        time.sleep(work_time)
                        self.outcome = StateOutcome.GPUCompileSuccess_CheckSuccess
                        self.eval_result[self.current_k] = f"Mock eval ran for {work_time}"

                    else:
                        start_time = time.time()

                        result = evaluate_single_sample_src(
                            ref_arch_src=self.ref_arch_src,
                            kernel_src=self.kernel_code[self.current_k],
                            configs=self.config,
                            build_dir=self.build_dir,
                            timeout_seconds=self.correct_state_timeout,
                            device=device
                        )
                        print(f"[Worker ({self.process_id}) working on problem {self.problem_id} sample {self.sample_id}] Result: ", result)

                        # pseudo code
                        # if result.correctness:
                            # we call profiler and record it to some field we have
                        
                        self.eval_result[self.current_k] = result
                        work_time = time.time() - start_time

                        if result is not None:
                            if "cuda_error" in result.metadata:
                                print(f"[Worker] CUDA Error detected, killing process {self.process_id} working on GPU {gpu_id}: Working on Problem {self.problem_id} Sample {self.sample_id}")
                                self.outcome = StateOutcome.Finish
                                return

                            if result.compiled:
                                if result.correctness:
                                    # Enable PyTorch profiling 
                                
                                    if "profiler" in self.config.context_strategy:
                                        print(
                                            f"[Worker] Process {self.process_id} using PyTorch profiler to profile on GPU {gpu_id} ({device})"
                                        )
                                        profile = get_torch_profiler_info(
                                            ref_arch_src=self.ref_arch_src,
                                            kernel_src=self.kernel_code[self.current_k],
                                            build_dir=self.build_dir,
                                            device=device,
                                        )
                                        print(profile)
                                        self.profiler_result[self.current_k] = profile
                                    self.outcome = StateOutcome.GPUCompileSuccess_CheckSuccess
                                else:
                                    self.outcome = StateOutcome.GPUCompileSuccess_CheckFail
                            else:
                                self.outcome = StateOutcome.GPUCompileFail
                        else:
                            self.outcome = StateOutcome.GPUCompileFail

                        print(
                            f"[Worker] Process {self.process_id} working on GPU {gpu_id} ({device}) for {work_time:.2f} seconds"
                        )
                        ###############################

                except TimeoutError:
                    print(f"[Worker] Process {self.process_id} GPU operation timed out")
                    self.outcome = StateOutcome.GPUCompileFail
                    self.eval_result[self.current_k] = kernel_eval.KernelExecResult(
                        compiled=False,
                        correctness=False,
                        metadata={"timeout_error": "GPU timed out.",
                                "hardware": f"gpu: {gpu_id}",
                                "device": f"gpu"
                                }
                        )

                except Exception as e:
                    print(f"[Worker] Process {self.process_id} encountered error: {e}")

                finally:
                    try:
                        del device
                    except:
                        pass
                    print(
                        f"[Worker] Process {self.process_id} released GPU {gpu_id}"
                    )

        else:
            print(f"[Single Worker] Process {self.process_id} requesting GPU 0...")
            device = torch.device(f"cuda:{self.config.dedicated_gpu_id}")
            print(f"[Single Worker] Acquired device:", device)
            if self.config.mock:
                # Simulate some GPU computation
                work_time = random.uniform(1, 5)
                torch.randn(100).to(device)
                time.sleep(work_time)
                self.eval_result[self.current_k] = f"Mock eval ran for {work_time}"
                self.outcome = StateOutcome.GPUCompileSuccess_CheckSuccess
            else:
                result = evaluate_single_sample_src(
                    ref_arch_src=self.ref_arch_src,
                    kernel_src=self.kernel_code[self.current_k],
                    configs=self.config,
                    build_dir=self.build_dir,
                    device=device
                )
                self.eval_result[self.current_k] = result

                if result is not None:
                    if result.compiled:
                        if result.correctness:
                            if "profiler" in self.config.context_strategy:
                                profile = get_torch_profiler_info(
                                    ref_arch_src=self.ref_arch_src,
                                    kernel_src=self.kernel_code[self.current_k],
                                    build_dir=self.build_dir,
                                    device=device,
                                )
                                self.profiler_result[self.current_k] = profile
                            self.outcome = StateOutcome.GPUCompileSuccess_CheckSuccess
                        else:
                            self.outcome = StateOutcome.GPUCompileSuccess_CheckFail
                    else:
                        self.outcome = StateOutcome.GPUCompileFail
                else:
                    self.outcome = StateOutcome.GPUCompileFail
                work_time = time.time() - start_time
            print(
                f"[Single Worker] Process working on GPU 0 for {work_time:.2f} seconds"
            )
            print(f"[Single Worker] Releasing GPU 0...")


        if finish:
            self.outcome = StateOutcome.Finish
        else:
            # For handling timeouts, now save
            self.logger.log_on_turn("eval_result", self.eval_result[self.current_k])

    def performance_logic(self):
        """
        Logic for the performance state. This is for profiling code.
        """
        show_current_state(self.current_k, self.state, self.config.show_state)


        if self.orchestrator is not None:
            print(f"[Worker] Process {self.process_id} requesting GPU...")
            
            with self.orchestrator.reserve_gpu() as gpu_id:
                try:
                    print(f"[Worker] Process {self.process_id} acquired GPU {gpu_id}")
                    # Put Eval Code Here
                    device = torch.device(f"cuda:{gpu_id}")
                
                    if self.config.mock:
                        return
                    else:
                        start_time = time.time()

                        result = get_torch_profiler_info(
                            ref_arch_src=self.ref_arch_src,
                            kernel_src=self.kernel_code[self.current_k],
                            build_dir=self.build_dir,
                            device=device,
                        )
                        self.profiler_result[self.current_k] = result
                        work_time = time.time() - start_time

                        print(
                            f"[Worker] Process {self.process_id} working on GPU {gpu_id} ({device}) for {work_time:.2f} seconds"
                        )
                        ###############################

                except Exception as e:
                    print(f"[Worker] Process {self.process_id} encountered error: {e}")

                finally:
                    try:
                        del device
                    except:
                        pass
                    print(
                        f"[Worker] Process {self.process_id} released GPU {gpu_id}"
                    )

        else:
            print(f"[Single Worker] Process {self.process_id} requesting GPU 0...")
            device = torch.device(f"cuda:{self.config.dedicated_gpu_id}")
            print(f"[Single Worker] Acquired device:", device)
            if self.config.mock:
                return
            else:
                result = get_torch_profiler_info(
                    ref_arch_src=self.ref_arch_src,
                    kernel_src=self.kernel_code[self.current_k],
                    build_dir=self.build_dir,
                    device=device,
                )
                self.profiler_result[self.current_k] = result
                work_time = time.time() - start_time
            print(
                f"[Single Worker] Process working on GPU 0 for {work_time:.2f} seconds"
            )
            print(f"[Single Worker] Releasing GPU 0...")

        self.outcome = StateOutcome.PerformanceSuccess

    def finish_logic(self):
        """
        Logic for the finish state.
        """
        show_current_state(self.current_k, self.state, self.config.show_state)
        self.outcome = StateOutcome.Finish

        # Log results as last run if applicable.
        if self.current_k > self.max_k:
            # Final eval step, this only happens in the last round 
            assert self.current_k == self.max_k + 1, "Final eval step should only happen in the last round"

            if self.current_k > 1:
                self.kernel_code[self.current_k] = self.kernel_code[self.current_k - 1]

                # VERY SPECIAL CASE: timeout at last step or memory error
                can_run = False if ("timeout_error" in self.eval_result[self.current_k - 1].metadata) else True

                if can_run:
                    self.compile_logic(finish=True)
                    self.correct_logic(finish=True)
                else:
                    # Propagate eval and compile results if it timed out earlier
                    self.feedback[self.current_k] = self.feedback[self.current_k - 1]
                    self.eval_result[self.current_k] = self.eval_result[self.current_k - 1]

                self.logger.log_turn(
                    self.current_k,
                    self.context.get(self.current_k-1, ""),
                    self.model_response.get(self.current_k-1, ""),
                    # Compile + Correctness feedback is Recent
                    self.kernel_code.get(self.current_k, ""),
                    self.feedback.get(self.current_k, ""),
                    self.eval_result.get(self.current_k, ""),
                    last=True,
                )
        
            # Mark that this run is finished:
            with open(os.path.join(self.log_dir, "DONE"), "w") as f:
                pass

        elif self.current_k <= self.max_k:
            "CUDA Error or other termination case."
            self.logger.log_turn(
                self.current_k,
                self.context.get(self.current_k, ""),
                self.model_response.get(self.current_k, ""),
                # Compile + Correctness feedback is Recent
                self.kernel_code.get(self.current_k, ""),
                self.feedback.get(self.current_k, ""),
                self.eval_result.get(self.current_k, ""),
            )
