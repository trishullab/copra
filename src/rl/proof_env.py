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
from src.tools.proof_exec_callback import ProofExecutorCallback
from src.rl.abstraction import State, Action, Env


class ProofEnv(Env):
    def __init__(self, 
        name, 
        dynamic_proof_executor_callback: ProofExecutorCallback,
        lemma_name: str,
        max_proof_depth: int = 10):
        assert isinstance(dynamic_proof_executor_callback, ProofExecutorCallback)
        assert isinstance(lemma_name, str)
        self.dynamic_proof_executor_callback = dynamic_proof_executor_callback
        self._dynamic_proof_executor = None
        self._loaded = False
        self._history : typing.List[typing.Tuple[State, Action, float, bool, dict]] = []
        self._name = name
        self.max_proof_depth = max_proof_depth
        self.lemma_name = lemma_name
        self._current_proof_depth = 0
        self._p_tree = ProofTree()
        pass

    @property
    def name(self):
        return self._name
    
    @property
    def state(self):
        assert self._loaded, "Env not loaded, call reset() first"
        current_goals = self._dynamic_proof_executor.get_current_proof_state_as_training_data()
        current_goals = copy.deepcopy(current_goals)
        state = ProofState(current_goals)
        return state
    
    @property
    def done(self) -> bool:
        assert self._loaded, "Env not loaded, call reset() first"
        needs_qed = self._dynamic_proof_executor.needs_qed()
        return needs_qed

    def reset(self):
        self._current_proof_depth = 0
        self._dynamic_proof_executor = self.dynamic_proof_executor_callback.get_proof_executor()
        self._history.clear()
        self._loaded = True
        pass

    def step(self, action: Action) -> typing.Tuple[State, float, bool, dict]:
        assert self._loaded, "Env not loaded, call reset() first"
        if self.done:
            return self.state, 0.0, True, {"Progress": "Done"}
        assert isinstance(action, ProofAction), f"action must be of type ProofAction, not {type(action)}"
        history_idx = len(self._history)
        state_before = self.state
        self._history.append((state_before, action, 0.0, False, {"Progress": "Starting"}))
        if action.action_type == ProofAction.Type.RUN_TACTIC:
            self._run_tactic(history_idx)
        pass

    def _run_tactic(self, history_idx: int = None):
        history_idx = len(self._history) - 1 if history_idx is None else history_idx
        state, action, reward, done, env_info = self._history[history_idx]
        assert isinstance(action, ProofAction)
        assert isinstance(state, ProofState)
        assert action.action_type == ProofAction.Type.RUN_TACTIC, "Action must be of type RUN_TACTIC"
        tactics = action.kwargs["tactics"]
        assert isinstance(tactics, list)
        assert len(tactics) > 0
        assert all([isinstance(tactic, str) for tactic in tactics])
        tactic_line_num, ran_successfully = self._dynamic_proof_executor.run_tactics(tactics)
        if ran_successfully:
            proof_progressed = self._current_proof_depth < self.max_proof_depth
            if proof_progressed:
                previous_proof_state = copy.deepcopy(state)
                # self.logger.info(f"Goal: {previous_proof_state.start_goals[:1]}")
                # self.logger.info(f"Tacitc: {next_tactic.proof_steps}")
                # self.logger.info(f"NextGoal: {current_proof_state.start_goals[:1]}")
                previous_proof_state.training_data_format.proof_steps = copy.deepcopy(tactics)
                current_proof_state = self.state
                # add the proof step to the proof tree
                # Check if the current proof state is less harder than the previous proof state
                if current_proof_state >= previous_proof_state:
                    # This is a cycle. Take a step back
                    next_step_add = False
                else:
                    next_step_add = p_tree.try_add_tactic(tactic_line_num, previous_proof_state)
                if not next_step_add:
                    proof_progressed = False
                    # self.logger.info(f"Got a cycle. Taking a step back.")
                else:
                    current_depth += 1
        else:
            proof_progressed = False
        pass