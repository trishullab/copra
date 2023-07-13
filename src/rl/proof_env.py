#!/usr/bin/env python3

import sys
root_dir = f"{__file__.split('src')[0]}"
if root_dir not in sys.path:
    sys.path.append(root_dir)
import copy
import typing
from src.rl.proof_tree import ProofTree
from src.rl.proof_state import ProofState
from src.rl.proof_action import ProofAction
from src.rl.abstraction import State, Action, Env
from src.tools.proof_exec_callback import ProofExecutorCallback
from src.tools.dynamic_proof_exec import DynamicProofExecutor
from dataclasses import dataclass
from dataclasses_json import dataclass_json


class ProgressState:
    STARTING = "Starting"
    RUNNING = "Running"
    DONE = "Done"
    FAILED = "Failed"
    def __init__(self):
        pass

@dataclass_json
@dataclass
class ProofEnvInfo(object):
    progress: str = ProgressState.STARTING
    error_message: typing.Optional[str] = None
    pass



class ProofEnv(Env):
    max_depth_penalty = -0.1
    max_proof_completion_reward = 1.0
    progress_reward = 0.2
    def __init__(self, 
        name: str, 
        dynamic_proof_executor_callback: ProofExecutorCallback,
        lemma_name: str,
        max_proof_depth: int = 10):
        assert isinstance(dynamic_proof_executor_callback, ProofExecutorCallback)
        assert isinstance(lemma_name, str)
        self.dynamic_proof_executor_callback = dynamic_proof_executor_callback
        self._dynamic_proof_executor = None
        self._loaded = False
        self._history : typing.List[typing.Tuple[State, Action, State, float, bool, ProofEnvInfo]] = []
        self._name = name
        self.max_proof_depth = max_proof_depth
        self.lemma_name = lemma_name
        self.current_proof_depth = 0
        self._p_tree = ProofTree()
        self._possible_failure_paths = 0
        self._success_path_length = 0
        self._num_cycles = 0
        pass

    def __enter__(self):
        self.reset()
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        if self._dynamic_proof_executor is not None:
            self._dynamic_proof_executor.__exit__(exc_type, exc_value, traceback)
        pass

    @property
    def name(self):
        return self._name
    
    @property
    def state(self):
        assert self._loaded, "Env not loaded, call reset() first"
        current_goals = self._dynamic_proof_executor.get_current_proof_state_as_training_data(DynamicProofExecutor.ContextType.NoContext)
        current_goals = copy.deepcopy(current_goals)
        state = ProofState(current_goals)
        return state
    
    @property
    def done(self) -> bool:
        assert self._loaded, "Env not loaded, call reset() first"
        needs_qed = self._dynamic_proof_executor.needs_qed()
        return needs_qed

    def reset(self):
        self.current_proof_depth = 0
        if self._dynamic_proof_executor is not None:
            self._dynamic_proof_executor.__exit__(None, None, None)
        self._dynamic_proof_executor = self.dynamic_proof_executor_callback.get_proof_executor()
        self._dynamic_proof_executor.__enter__()
        self._history.clear()
        self._loaded = True
        self._foward_to_lemma_proof()
        pass

    def step(self, action: Action) -> typing.Tuple[State, float, bool, dict]:
        assert self._loaded, "Env not loaded, call reset() first"
        info = ProofEnvInfo(progress=ProgressState.STARTING)
        if self.done:
            info.progress = ProgressState.DONE
            return self.state, 0.0, True, info.to_dict()
        assert isinstance(action, ProofAction), f"action must be of type ProofAction, not {type(action)}"
        history_idx = len(self._history)
        state_before = self.state
        self._history.append((state_before, action, None, 0.0, False, info))
        if action.action_type == ProofAction.Type.RUN_TACTIC:
            self._run_tactic(history_idx)
        pass
        return self._history[-1][2], self._history[-1][3], self._history[-1][4], self._history[-1][5].to_dict()
    
    def checkpoint(self):
        return super().checkpoint()
    
    def clone(self):
        return super().clone()
    
    def render(self):
        return super().render()

    def _run_tactic(self, history_idx: int = None):
        history_idx = len(self._history) - 1 if history_idx is None else history_idx
        state, action, current_proof_state, reward, done, env_info = self._history[history_idx]
        assert isinstance(action, ProofAction)
        assert isinstance(state, ProofState)
        assert action.action_type == ProofAction.Type.RUN_TACTIC, "Action must be of type RUN_TACTIC"
        tactics = action.kwargs["tactics"]
        assert isinstance(tactics, list)
        assert len(tactics) > 0
        assert all([isinstance(tactic, str) for tactic in tactics])
        tactic_line_num, ran_successfully = self._dynamic_proof_executor.run_tactics(tactics)
        cycle_detected = False
        proof_progressed = False
        if ran_successfully:
            previous_proof_state = copy.deepcopy(state)
            previous_proof_state.training_data_format.proof_steps = copy.deepcopy(tactics)
            current_proof_state = self.state
            # add the proof step to the proof tree
            # Check if the current proof state is less harder than the previous proof state
            if current_proof_state >= previous_proof_state:
                # This is a cycle. Take a step back
                next_step_add = False
            else:
                next_step_add = self._p_tree.try_add_tactic(tactic_line_num, previous_proof_state)
            if not next_step_add:
                proof_progressed = False
                cycle_detected = True
                # self.logger.info(f"Got a cycle. Taking a step back.")
            else:
                proof_progressed = True
                self.current_proof_depth += len(tactics)
        else:
            proof_progressed = False
        if not proof_progressed:
            self._possible_failure_paths += 1
            assert len(self._p_tree) == self.current_proof_depth, "proof_tree must have the same length as current_depth"
            # cancel anything which might got executed
            if self._dynamic_proof_executor.cancel_tactic_till_line(tactic_line_num): # This becomes relevant when we have a cycle
                assert cycle_detected, "cycle_detected must be true if cancel_tactic_till_line returns true"
                self._num_cycles += 1
        assert (proof_progressed and not cycle_detected) or (not proof_progressed and cycle_detected) or (not proof_progressed and not cycle_detected), "proof_progressed and cycle_detected cannot be true at the same time"
        reward = 0.0
        depth_ratio = self.current_proof_depth/self.max_proof_depth
        if depth_ratio > 1.0:
            depth_ratio = 1.0
        depth_penalty = depth_ratio * ProofEnv.max_depth_penalty
        reward += depth_penalty
        done = self.done
        if proof_progressed and done:
            reward += ProofEnv.max_proof_completion_reward
            env_info.progress = ProgressState.DONE
            env_info.error_message = None
        elif proof_progressed:
            reward += ProofEnv.progress_reward
            env_info.progress = ProgressState.RUNNING
            env_info.error_message = None
        elif cycle_detected:
            env_info.progress = ProgressState.FAILED
            env_info.error_message = "The proof steps taken until now resulted in a cycle i.e. same goal repeated after some steps. Try taking a step back and try again."
            current_proof_state = state
        else:
            env_info.progress = ProgressState.FAILED
            env_info.error_message = self._dynamic_proof_executor.get_last_exception()
            current_proof_state = state
        self._history[history_idx] = (state, action, current_proof_state, reward, done, env_info)
        pass

    def _foward_to_lemma_proof(self):
        assert self._loaded, "Env not loaded, call reset() first"
        lemma_found = False
        while not self._dynamic_proof_executor.execution_complete and not lemma_found:
            assert not self._dynamic_proof_executor.is_in_proof_mode(), "executor must not be in proof mode"
            _ = list(self._dynamic_proof_executor.run_till_next_lemma_return_exec_stmt())
            if self._dynamic_proof_executor.execution_complete:
                break
            lemma_name_with_stmt = self._dynamic_proof_executor.get_lemma_name_if_running().strip()
            lemma_found = lemma_name_with_stmt.startswith(self.lemma_name)
            if not lemma_found:
                _ = list(self._dynamic_proof_executor.run_to_finish_lemma_return_exec())
                if self._dynamic_proof_executor.execution_complete:
                    break

        if not lemma_found:
            raise Exception(f"Could not find lemma {self.lemma_name}")
        pass


if __name__ == "__main__":
    import os
    os.chdir(root_dir)
    print("Interactive Proof Environment")
    proof_exec_callback = ProofExecutorCallback(
        project_folder=".",
        file_path="data/test/SimpleAlgebra.v"
    )
    with ProofEnv("test", proof_exec_callback, 'algb_add_comm', max_proof_depth=10) as env:
        done = env.done
        print(f"Starting state: \n{env.state.serialize()}")
        inp = input("Enter a tactic: ")
        while inp != "exit" and not done:
            action = ProofAction(ProofAction.Type.RUN_TACTIC, tactics=[inp])
            state, reward, done, info = env.step(action)
            print(f"Reward: {reward}")
            print(f"Info: \n{info}")
            print(f"State: \n{state.serialize()}")
            if not done:
                inp = input("Enter a tactic: ")
        pass