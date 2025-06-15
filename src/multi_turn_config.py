import os
import pydra
from pydra import Config, REQUIRED

REPO_TOP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class MultiTurnConfig(Config):
    def __init__(self):

        self.dataset_src = REQUIRED
        self.dataset_name = "ScalingIntelligence/KernelBench"

        self.level = 1
        self.problem_id = 1
        # This is used in legacy caesar code to define the prompt id, in newer KernelBench it should be example_ind
        self.num_samples = 1  

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
        self.num_gpus = 1
        self.num_workers = 1

        self.log_dir_prefix = os.path.join(REPO_TOP_DIR, "../results/kernel_multi_turn")
        self.build_dir_prefix = os.path.join(REPO_TOP_DIR, "../results/kernel_eval_build")

        self.run_group = "dummy_run_group"
        self.run_name = "dummy_run_name"

        self.verbose = True
        self.show_state = True
        self.measure_performance = True
        self.mock = True
        self.debug = False

        # multi-turn numbers
        self.max_k = 10
        self.context_strategy = ["reflection"]
        self.state_machine_strategy = "rerun" # default
        self.max_feedback_length = 100000 # in terms of characters, 10k, so much less in tokens
        # TODO: this is not implemented in the state machine yet
        self.use_last_only = False 

        def qwen2_5_coder_32b_instruct():
            self.model_name = "Qwen/Qwen2.5-Coder-32B-Instruct"
            self.greedy_sample = False
            self.temperature = 0.7
            self.top_p = 0.8
            self.top_k = 20
            self.num_completions = 1
            self.max_tokens = 24576