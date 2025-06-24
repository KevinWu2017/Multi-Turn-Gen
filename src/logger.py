"""
Logger class for the Caesar state machine that saves progress and results to JSON files.
"""

import json
import os
from typing import Dict, Any, Optional
from pathlib import Path
from dataclasses import asdict
from pydra import Config

from states import WorkArgs
from utils import ensure_json_serializable

from src import eval as kernel_eval

class CaesarLogger:
    def __init__(
            self, 
            log_dir: str, 
            config: Config,
            work: WorkArgs,
            verbose: bool = False, 
            log_name="log.json"
        ):
        """
        Initialize the logger with a directory to save logs.
        
        Args:
            log_dir: Directory path where logs will be saved
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / log_name
        self.verbose = verbose
        self.turn = 1
        
        self.current_log = {}

        self.setup_logging_dir(config.to_dict())

        metadata = asdict(work)
        metadata["run_name"] = config.run_name
        metadata["num_rounds"] = config.max_k
        self.log_metadata(metadata)

    def setup_logging_dir(self, config: dict) -> None:
        config_path = self.log_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(ensure_json_serializable(config), f, indent=2)

    def log_metadata(self, info: dict) -> None:
        """
        Log metadata. May assert that info has
        the right keys.
        """
        required_keys = ["run_name", "num_rounds"]
        for key in required_keys: assert(key in info)

        self.current_log["metadata"] = info

    def update_turn(self, turn: int) -> None:
        self.turn = turn
        self.current_log[self.turn] = {}
        self._save_log()
    
    def clean_log(self) -> None:
        """
        Hacky: Only call when finished!
        """
        if self.current_log[self.turn] == {}:
            del self.current_log[self.turn]
        self._save_log()

    def log_on_turn(self, key: str, content):
        """
        Hacky function to add to current turn.
        """
        if self.turn in self.current_log:
            self.current_log[self.turn][key] = content
        # self._save_log()

    def flush_log_now(self):
        self._save_log()

    def log_turn(self, 
                turn: int,
                context: Optional[str] = None,
                model_reasoning_response: Optional[str] = None,
                model_response: Optional[str] = None,
                kernel_code: Optional[str] = None,
                feedback: Optional[str] = None,
                eval_result: Optional[str] = None,
                profiler_result: Optional[str] = None,
                last: bool = False
        ) -> None:
        """
        Log the data for a specific turn.
        
        Args:
            turn: Turn number to log
            context: Optional context string for this turn
            model_response: Optional model response string for this turn
            kernel_code: Optional kernel code string for this turn
            feedback: Optional feedback string for this turn
        """
        if last: turn = "result" # Python checking is fine

        if turn not in self.current_log:
            self.current_log[turn] = {
                "context": "",
                "model_response": "",
                "kernel_code": "",
                "feedback": "",
                "eval_result": "",
                "profiler_result": "",
            }

        if context is not None:
            self.current_log[turn]["context"] = context
        if model_response is not None:
            self.current_log[turn]["model_response"] = model_response
        if kernel_code is not None:
            self.current_log[turn]["kernel_code"] = kernel_code
        if feedback is not None:
            self.current_log[turn]["feedback"] = feedback
        if eval_result is not None:
            self.current_log[turn]["eval_result"] = eval_result
        if profiler_result is not None:
            self.current_log[turn]["profiler_result"] = profiler_result
            
        if self.verbose:
            print(f"Saved turn {turn} info to {self.log_file}")

            dump_file = os.path.join(self.log_dir, f"context_dump_{turn}.txt")
            with open(dump_file, 'w') as f:
                f.write(context)

        self._save_log()

    def _save_log(self) -> None:
        """Save the current log to a JSON file."""
        with open(self.log_file, 'w') as f:
            json.dump(ensure_json_serializable(self.current_log), f, indent=2)

    def _load_log(self) -> None:
        """Load existing log data from the log file if it exists."""
        if self.log_file.exists():
            with open(self.log_file, 'r') as f:
                self.current_log = json.load(f)
                # Find highest numeric turn
                numeric_turns = [int(k) for k in self.current_log.keys() if str(k).isdigit()]

                for turn in numeric_turns:
                    if "eval_result" in self.current_log[turn]:
                        self.current_log[turn] = self.exec_log_to_obj(self.current_log[turn])

                self.turn = max(numeric_turns) if numeric_turns else 1
    
    def exec_log_to_obj(self, saved_dict: dict | str) -> kernel_eval.KernelExecResult | None:
        if isinstance(saved_dict, kernel_eval.KernelExecResult):
            return saved_dict

        if saved_dict == "": 
            # return None
            # SHOULD NOT HAPPEN
            raise ValueError("[Logger] exec_log_to_obj: saved_dict is empty, should not happen")
        
        kernel_eval_result = kernel_eval.KernelExecResult(
            compiled=saved_dict.get("compiled", None),
            correctness=saved_dict.get("correctness", None),
            metadata=saved_dict.get("metadata", {}),
            runtime=saved_dict.get("runtime", -1.0),
            runtime_stats=saved_dict.get("runtime_stats", {})
        )        
        return kernel_eval_result
    

    def _save_timeout_eval(self, msg: str) -> None:
        """
        Special case for handling eval timeouts on GPU.
        """
        self._load_log()
        self.current_log[str(self.turn)]["eval_result"] = kernel_eval.KernelExecResult(
                compiled=False,
                correctness=False,
                metadata={"timeout_error": msg,
                          "hardware": "", # TODO: if necessary, add
                          "device": ""
                        }
                )

        self._save_log()

    def get_turn_data(self, turn: int) -> Dict[str, str]:
        """
        Get the logged data for a specific turn.
        
        Args:
            turn: Turn number to retrieve
            
        Returns:
            Dictionary containing the turn's logged data
        """
        return self.current_log.get(turn)
