from enum import Enum
from typing import Dict
from abc import ABC, abstractmethod
from dataclasses import dataclass

@dataclass
class WorkArgs:
    """What the state machine is going to run"""
    problem: str
    problem_id: str
    sample_id: int
    

class CaesarState(Enum):
    """State machine states"""
    START_STATE = "start"
    GENERATE_STATE = "generate" 
    COMPILE_STATE = "compile" 
    CORRECT_STATE = "correct"
    PERFORMANCE_STATE = "performance"
    FINISH_CHECK_STATE = "finish_check"
    FINISH_STATE = "finish"

class StateOutcome(Enum):
    # These outcomes are specific to ONE state
    """Possible outcomes at a state"""
    # Outcomes for START_STATE
    Start = "start"

    # Outcomes for GENERATE_STATE
    Generate = "generate"

    # Outcomes for COMPILE_STATE
    CPUCompileSuccess = "cpu_compile_success"
    CPUCompileFail = "cpu_compile_fail"

    # Outcomes for CORRECT_STATE
    GPUCompileFail = "gpu_compile_fail"
    GPUCompileSuccess_CheckFail = "gpu_compile_success_check_fail"
    GPUCompileSuccess_CheckSuccess = "gpu_compile_success_check_success"
    
    # Outcomes for PERFORMANCE_STATE
    PerformanceSuccess = "performance_success"
    PerformanceFail = "performance_fail"

    # FINISH_STATE
    Finish = "finish"

class Transition(ABC, Dict[StateOutcome, CaesarState]):
    """
    Abstract base class that maps (current_state, outcome) pairs to next states.
    Subclasses must implement the transitions for all valid state-outcome combinations.
    """
    
    def __init__(self):
        super().__init__()
        self._validate_transitions()
    
    @abstractmethod
    def _define_transitions(self) -> Dict[StateOutcome, CaesarState]:
        """
        Define the mapping of (current_state, outcome) pairs to next states.
        Must be implemented by subclasses.
        """
        pass
    
    def _validate_transitions(self) -> None:
        """
        Ensures all valid state-outcome combinations have a defined transition
        and loads them into the dictionary.
        for now let's check all the outcomes are defined
        """
        # TODO:
        transitions = self._define_transitions()

        # check all the outcomes are defined
        for outcome in StateOutcome:
            if outcome not in transitions:
                raise ValueError(f"Mapping for Outcome {outcome} is not defined")

        self.update(transitions) # updates the dict
        