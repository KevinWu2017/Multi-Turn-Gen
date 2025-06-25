import os
import pydra
from pydra import Config, REQUIRED

REPO_TOP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# TODO: shoule we minus the max_tokens by the system prompt length?
# a list of presets for API server configs
SERVER_PRESETS = {
    "deepseek": {
        "temperature": 0.0,
        "model_name": "deepseek-reasoner",
        "max_tokens": 64*1024,  # 64k tokens
        "server_type": "deepseek",
        "is_reasoning_model": True,
    },
    "together": {  # mostly for Llama 3.1
        "model_name": "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
        # "model_name": "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
        "temperature": 0.7,
        "max_tokens": 4096,
    },
    "local-deploy": { # for local deployment, like vllm
        "server_type": "local-deploy",
        "server_address": "10.0.16.46",
        "server_port": 8001,
        "model_name": "Qwen/Qwen3-32B",
        "is_reasoning_model": True,
        "max_tokens": 32768,
        "temperature": 0.0,
        "top_p": 0.95,  # nucleus sampling
        "top_k": 20,  # top-k sampling
        "max_k": 50,  # max number of turns
    },
}


class MultiTurnConfig(Config):
    def __init__(self):

        self.dataset_src = "local"
        self.dataset_name = "ScalingIntelligence/KernelBench"

        self.level = 1
        self.problem_id = 1
        # This is used in legacy caesar code to define the prompt id, in newer KernelBench it should be example_ind
        self.num_samples = 1
        self.use_gpu_info = False

        # LLM configs
        self.server_type = None
        self.server_address = None
        self.server_port = None
        self.model_name = None
        self.is_reasoning_model = False

        # decoding parameter
        self.greedy_sample = False
        self.temperature = 0.0
        self.top_p = 1.0  # set to consider all tokens
        self.top_k = 50  # set large for default
        self.num_completions = 1
        self.max_tokens = 32768

        # Eval Speciifc
        self.num_correct_trials = 5
        self.num_perf_trials = 100
        self.timeout = 600  # time out per round, set to 10 min
        self.gpu_name = "A100"
        self.gpu_arch = ["Ampere"]
        self.num_gpus = 1
        self.num_workers = 1

        self.log_dir_prefix = os.path.join(REPO_TOP_DIR, "results/kernel_multi_turn")
        self.build_dir_prefix = os.path.join(REPO_TOP_DIR, "results/kernel_eval_build")

        self.run_group = "dummy_run_group"
        self.run_name = "dummy_run_name"

        self.print_inference_output = True
        self.stream_inference = True
        self.verbose = True
        self.show_state = True
        self.measure_performance = True
        self.dry_run = False
        self.debug = False
        self.simulate_error = True 
        self.simulate_error_type = 'none' # 'none', 'compile', 'correctness', 'cuda_error'

        # multi-turn numbers
        self.max_k = 10
        self.context_strategy = ["reflection"]
        self.state_machine_strategy = "rerun"  # default
        self.max_feedback_length = (
            100000  # in terms of characters, 10k, so much less in tokens
        )
        self.use_last_only = False

    def set_preset_config(self, preset_name):
        """
        根据预设名称设置配置参数

        Args:
            preset_name (str): 预设配置的名称，可选值为 SERVER_PRESETS 中的键
                             如: "deepseek", "together", "sglang"

        Raises:
            ValueError: 当提供的预设名称不存在时抛出异常
        """
        if preset_name not in SERVER_PRESETS:
            available_presets = list(SERVER_PRESETS.keys())
            raise ValueError(
                f"未知的预设名称 '{preset_name}'。可用的预设有: {available_presets}"
            )

        preset_config = SERVER_PRESETS[preset_name]

        # 应用预设配置中的所有参数到当前实例
        for key, value in preset_config.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                print(f"警告: 预设配置中的参数 '{key}' 在配置类中不存在，已跳过")

        print(f"已成功应用预设配置 '{preset_name}'")
