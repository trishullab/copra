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
from src.retrieval.coq_bm25_reranker import CoqBm25ReRanker
from dataclasses import dataclass
from dataclasses_json import dataclass_json
from enum import Enum


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

class ProofEnvReRankStrategy(Enum):
    BM25 = 1

class ProofEnv(Env):
    max_depth_penalty = -0.1
    max_proof_completion_reward = 1.0
    progress_reward = 0.2
    def __init__(self, 
        name: str, 
        dynamic_proof_executor_callback: ProofExecutorCallback,
        lemma_name: str,
        retrieval_strategy: ProofEnvReRankStrategy = ProofEnvReRankStrategy.BM25,
        max_proof_depth: int = 10):
        assert isinstance(dynamic_proof_executor_callback, ProofExecutorCallback)
        assert isinstance(lemma_name, str)
        self.dynamic_proof_executor_callback = dynamic_proof_executor_callback
        self._dynamic_proof_executor : DynamicProofExecutor = None
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
        self.retrieve_strategy = retrieval_strategy
        if self.retrieve_strategy == ProofEnvReRankStrategy.BM25:
            self._re_ranker = CoqBm25ReRanker()
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
        current_goals = self._dynamic_proof_executor.get_current_proof_state_as_training_data()
        current_goals = copy.deepcopy(current_goals)
        state = ProofState(current_goals)
        return state
    
    @property
    def done(self) -> bool:
        assert self._loaded, "Env not loaded, call reset() first"
        needs_qed = self._dynamic_proof_executor.needs_qed()
        return needs_qed

    @property
    def history(self) -> typing.List[typing.Tuple[State, Action, State, float, bool, ProofEnvInfo]]:
        assert self._loaded, "Env not loaded, call reset() first"
        return self._history

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

    def step(self, action: Action) -> typing.Tuple[State, Action, State, float, bool, ProofEnvInfo]:
        assert self._loaded, "Env not loaded, call reset() first"
        info = ProofEnvInfo(progress=ProgressState.STARTING)
        if self.done:
            info.progress = ProgressState.DONE
            return self.state, 0.0, True, info
        assert isinstance(action, ProofAction), f"action must be of type ProofAction, not {type(action)}"
        history_idx = len(self._history)
        state_before = self.state
        self._history.append((state_before, action, None, 0.0, False, info))
        if action.action_type == ProofAction.ActionType.RUN_TACTIC:
            self._run_tactic(history_idx)
        elif action.action_type == ProofAction.ActionType.GET_DFNS:
            self._get_dfns(history_idx)
        elif action.action_type == ProofAction.ActionType.GET_THMS:
            self._get_thms(history_idx)
        else:
            raise NotImplementedError(f"Action type {action.action_type} not implemented")
        return self._history[-1][0], self._history[-1][1], self._history[-1][2], self._history[-1][3], self._history[-1][4], self._history[-1][5]
    
    def checkpoint(self):
        return super().checkpoint()
    
    def clone(self):
        return super().clone()
    
    def render(self):
        return super().render()

    def _run_tactic(self, history_idx: int = None):
        assert self._loaded, "Env not loaded, call reset() first"
        history_idx = len(self._history) - 1 if history_idx is None else history_idx
        state, action, current_proof_state, reward, done, env_info = self._history[history_idx]
        assert isinstance(action, ProofAction)
        assert isinstance(state, ProofState)
        assert action.action_type == ProofAction.ActionType.RUN_TACTIC, "Action must be of type RUN_TACTIC"
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
                self.current_proof_depth += 1
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

    def _get_thms(self, history_idx: int = None):
        assert self._loaded, "Env not loaded, call reset() first"
        history_idx = len(self._history) - 1 if history_idx is None else history_idx
        state, action, current_proof_state, reward, done, env_info = self._history[history_idx]
        assert isinstance(action, ProofAction)
        assert isinstance(state, ProofState)
        assert action.action_type == ProofAction.ActionType.GET_THMS, "Action must be of type GET_THMS"
        relevant_thms = self._dynamic_proof_executor.get_all_relevant_thms()
        for goal in relevant_thms.start_goals:
            query = goal.goal
            local_responses = [str(relevant_thms.all_useful_defns_theorems[lemma_ref.lemma_idx]) for lemma_ref in goal.possible_useful_theorems_local]
            global_responses = [str(relevant_thms.all_useful_defns_theorems[lemma_ref.lemma_idx]) for lemma_ref in goal.possible_useful_theorems_external]
            local_scores = self._re_ranker.rerank(query, local_responses)
            global_scores = self._re_ranker.rerank(query, global_responses)
            local_idx = [(idx, score) for idx, score in enumerate(local_scores)]
            global_idx = [(idx, score) for idx, score in enumerate(global_scores)]
            local_idx.sort(key=lambda x: x[1], reverse=True)
            global_idx.sort(key=lambda x: x[1], reverse=True)
            local_responses = [goal.possible_useful_theorems_local[idx] for idx, _ in local_idx]
            global_responses = [goal.possible_useful_theorems_external[idx] for idx, _ in global_idx]
            sum_local_scores = sum([score for _, score in local_idx]) + 1e-6
            sum_global_scores = sum([score for _, score in global_idx]) + 1e-6
            for i in range(len(local_responses)):
                local_responses[i].score = local_idx[i][1]/sum_local_scores
            for i in range(len(global_responses)):
                global_responses[i].score = global_idx[i][1]/sum_global_scores
            goal.possible_useful_theorems_local = local_responses
            goal.possible_useful_theorems_external = global_responses
        current_proof_state = ProofState(relevant_thms)
        reward = 0.0
        done = self.done
        env_info.progress = ProgressState.RUNNING if not done else ProgressState.DONE
        env_info.error_message = None
        self._history[history_idx] = (state, action, current_proof_state, reward, done, env_info)
        pass

    def _get_dfns(self, history_idx: int = None):
        assert self._loaded, "Env not loaded, call reset() first"
        history_idx = len(self._history) - 1 if history_idx is None else history_idx
        state, action, current_proof_state, reward, done, env_info = self._history[history_idx]
        assert isinstance(action, ProofAction)
        assert isinstance(state, ProofState)
        assert action.action_type == ProofAction.ActionType.GET_DFNS, "Action must be of type GET_DEFNS"
        relevant_defns = self._dynamic_proof_executor.get_all_relevant_defns()
        for goal in relevant_defns.start_goals:
            query = goal.goal
            responses = [str(relevant_defns.all_useful_defns_theorems[lemma_ref.lemma_idx]) for lemma_ref in goal.relevant_defns]
            response_scores = self._re_ranker.rerank(query, responses)
            relevant_defns_idx = [(idx, score) for idx, score in enumerate(response_scores)]
            relevant_defns_idx.sort(key=lambda x: x[1], reverse=True)
            relevant_defns_reranked = [goal.relevant_defns[idx] for idx, _ in relevant_defns_idx]
            sum_scores = sum([score for _, score in relevant_defns_idx]) + 1e-6
            for i in range(len(relevant_defns_reranked)):
                relevant_defns_reranked[i].score = relevant_defns_idx[i][1]/sum_scores
            goal.relevant_defns = relevant_defns_reranked
        current_proof_state = ProofState(relevant_defns)
        reward = 0.0
        done = self.done
        env_info.progress = ProgressState.RUNNING if not done else ProgressState.DONE
        env_info.error_message = None
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
    supported_actions = [x.name for x in ProofAction.ActionType]

    def scan_action():
        inp_action_type = input(f"Enter an action type from {supported_actions}: ")
        action_type = ProofAction.ActionType[inp_action_type]
        if action_type == ProofAction.ActionType.RUN_TACTIC:
            inp = input("Enter tactic(s) (';' separated): ")
            inp = inp.split(';')
            return ProofAction(action_type, tactics=inp)
        elif action_type == ProofAction.ActionType.GET_THMS:
            return ProofAction(action_type)
        elif action_type == ProofAction.ActionType.GET_DFNS:
            return ProofAction(action_type)
        elif action_type == ProofAction.ActionType.EXIT:
            return ProofAction(action_type)
        else:
            raise Exception(f"Invalid action type {action_type}")
    

    with ProofEnv("test", proof_exec_callback, 'algb_add_comm', max_proof_depth=10) as env:
        done = env.done
        print(f"Starting state: \n{env.state.serialize()}")
        action = scan_action()
        while action.action_type != ProofAction.ActionType.EXIT and not done:
            state, _, _, reward, done, info = env.step(action)
            print(f"Reward: {reward}")
            print(f"Info: \n{info}")
            print(f"State: \n{state.serialize()}")
            if not done:
                action = scan_action()
        pass