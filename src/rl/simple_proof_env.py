#!/usr/bin/env python3

import sys
root_dir = f"{__file__.split('src')[0]}"
if root_dir not in sys.path:
    sys.path.append(root_dir)
import copy
import typing
import logging
import time
from src.rl.proof_tree import ProofSearchResult, ProofTree
from src.rl.proof_state import ProofState
from src.rl.proof_action import ProofAction
from src.rl.abstraction import State, Action, Env
from src.tools.proof_exec_callback import ProofExecutorCallback
from src.tools.dynamic_proof_exec import DynamicProofExecutor
from src.retrieval.coq_bm25_reranker import CoqBm25ReRanker
from dataclasses import dataclass, field
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
    info_messages: typing.List[str] = field(default_factory=list)
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
        max_proof_depth: int = 10,
        logger : logging.Logger = None):
        assert isinstance(dynamic_proof_executor_callback, ProofExecutorCallback)
        assert isinstance(lemma_name, str)
        self.dynamic_proof_executor_callback = dynamic_proof_executor_callback
        self._dynamic_proof_executor : DynamicProofExecutor = None
        self._loaded = False
        self._history : typing.List[typing.Tuple[ProofState, ProofAction, ProofState, float, bool, ProofEnvInfo]] = []
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
        else:
            raise NotImplementedError(f"Retrieval strategy {self.retrieve_strategy} not implemented")
        self.logger = logger if logger is not None else logging.getLogger(__name__)

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
        use_proof_executor_state = True
        # Just check the last action in history
        if len(self._history) > 0:
            _, action, s2, _, _, _ = self._history[-1]
            if action.action_type == ProofAction.ActionType.GET_DFNS or \
                action.action_type == ProofAction.ActionType.GET_THMS:
                use_proof_executor_state = False
                current_goals = s2
        if use_proof_executor_state:
            current_goals = self._dynamic_proof_executor.get_current_proof_state_as_training_data()
        current_goals = copy.deepcopy(current_goals)
        current_proof_tree = copy.deepcopy(self._p_tree)
        state = ProofState(current_goals)
        state.proof_tree = current_proof_tree
        return state
    
    @property
    def done(self) -> bool:
        assert self._loaded, "Env not loaded, call reset() first"
        # needs_qed = self._dynamic_proof_executor.needs_qed()
        not_in_proof_mode = not self._dynamic_proof_executor.is_in_proof_mode()
        # return needs_qed or not_in_proof_mode
        return not_in_proof_mode

    @property
    def history(self) -> typing.List[typing.Tuple[ProofState, ProofAction, ProofState, float, bool, ProofEnvInfo]]:
        assert self._loaded, "Env not loaded, call reset() first"
        return self._history

    def reset(self):
        self.current_proof_depth = 0
        if self._dynamic_proof_executor is not None:
            self._dynamic_proof_executor.__exit__(None, None, None)
        self._dynamic_proof_executor = self.dynamic_proof_executor_callback.get_proof_executor()
        self._dynamic_proof_executor.__enter__()
        self._history.clear()
        self._p_tree = ProofTree()
        self._loaded = True
        self._foward_to_lemma_proof()
        self.goal_start_time = time.time()
        self.inferences_used = 0
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
        elif action.action_type == ProofAction.ActionType.BACKTRACK:
            self._backtrack(history_idx)
        else:
            raise NotImplementedError(f"Action type {action.action_type} not implemented")
        self.inferences_used += 1
        return self._history[-1][0], self._history[-1][1], self._history[-1][2], self._history[-1][3], self._history[-1][4], self._history[-1][5]
    
    def checkpoint(self):
        return super().checkpoint()
    
    def clone(self):
        return super().clone()
    
    def render(self):
        s1, a, s2, r, d, info = self._history[-1]
        visibility = 3
        self.logger.info("-"*50)
        s1_relevant_dfns = [
            "\n".join([str(s1.training_data_format.all_useful_defns_theorems[dfns.lemma_idx]) for dfns in goal.relevant_defns]) 
        for goal in s1.training_data_format.start_goals]
        s1_possible_thms = [
                "\n".join([str(s1.training_data_format.all_useful_defns_theorems[thm.lemma_idx]) 
            for thm in (goal.possible_useful_theorems_local[:visibility] + goal.possible_useful_theorems_external[:visibility])])
        for goal in s1.training_data_format.start_goals]
        s1_goals = [f"Goal [{idx}]:\n {goal.goal} \n Hyps [{idx}]:\n {goal.hypotheses} \n Dfns [{idx}]:\n {s1_relevant_dfns[idx]} \n Thms [{idx}]:\n {s1_possible_thms[idx]} \n------------------\n" for idx, goal in enumerate(s1.training_data_format.start_goals)]
        s1_goal = '\n'.join(s1_goals)
        self.logger.info(f"Proof State (before action):\n {s1_goal}")
        s2_relevant_dfns = [
            "\n".join([str(s2.training_data_format.all_useful_defns_theorems[dfns.lemma_idx]) for dfns in goal.relevant_defns])
        for goal in s2.training_data_format.start_goals]
        s2_possible_thms = [
                "\n".join([str(s2.training_data_format.all_useful_defns_theorems[thm.lemma_idx]) 
            for thm in (goal.possible_useful_theorems_local[:visibility] + goal.possible_useful_theorems_external[:visibility])])
        for goal in s2.training_data_format.start_goals]
        s2_goals = [f"Goal [{idx}]:\n {goal.goal} \n Hyps [{idx}]: {goal.hypotheses} \n Dfns [{idx}]:\n {s2_relevant_dfns[idx]} \n Thms [{idx}]:\n {s2_possible_thms[idx]} \n-------------------\n" for idx, goal in enumerate(s2.training_data_format.start_goals)]
        action = a.serialize()
        self.logger.info(f"Action:\n {action}")
        s2_goal = '\n'.join(s2_goals)
        self.logger.info(f"Proof State (after action):\n {s2_goal}")
        self.logger.info(f"Reward:\n {r}")
        self.logger.info(f"Done:\n {d}")
        self.logger.info(f"Info:\n {info.to_json()}")
        self.logger.info("-"*50)
        pass

    def dump_proof(self):
        assert self._loaded, "Env not loaded, call reset() first"
        self.goal_end_time = time.time()
        self.time_taken = self.goal_end_time - self.goal_start_time
        self.proof_search_res = ProofSearchResult(
            self._dynamic_proof_executor.main_file, 
            not self._dynamic_proof_executor.is_in_proof_mode(), 
            self._lemma_name_with_stmt, 
            [tactic.training_data_format for _, tactic in self._p_tree], 
            self.time_taken, 
            self.inferences_used, 
            possible_failed_paths=-1, 
            num_of_backtracks=-1, 
            is_timeout=False, 
            is_inference_exhausted=False, 
            longest_success_path=-1)
        self.logger.info(f"Dumping proof search result:\n {self.proof_search_res}")

    def _run_tactic(self, history_idx: int = None):
        assert self._loaded, "Env not loaded, call reset() first"
        history_idx = len(self._history) - 1 if history_idx is None else history_idx
        state, action, _, reward, done, env_info = self._history[history_idx]
        # was_done_before = done
        assert action.action_type == ProofAction.ActionType.RUN_TACTIC, "Action must be of type RUN_TACTIC"
        tactics = action.kwargs["tactics"]
        assert isinstance(tactics, list)
        assert len(tactics) > 0
        assert all([isinstance(tactic, str) for tactic in tactics])
        # Remove unnecessary spaces, newlines, and tabs
        tactics = [tactic.strip() for tactic in tactics]
        state, next_state, reward, done, env_info = self._run_tactics(tactics, state, action, env_info)
        self._history[history_idx] = (state, action, next_state, reward, done, env_info)

    
    def _run_tactics(self, tactics: typing.List[str], state: ProofState, action: ProofAction, env_info: ProofEnvInfo):
        env_info = copy.deepcopy(env_info)
        tactic_line_num, ran_successfully = self._dynamic_proof_executor.run_tactics(tactics)
        proof_progressed = False
        if ran_successfully:
            previous_proof_state = state
            previous_proof_state.training_data_format.proof_steps = copy.deepcopy(tactics)
            current_proof_state = self.state
            # add the proof step to the proof tree
            self._p_tree.try_add_tactic(tactic_line_num, previous_proof_state, force_add=True, action=action)
            proof_progressed = True
            self.current_proof_depth += 1
        else:
            proof_progressed = False
        if not proof_progressed:
            self._possible_failure_paths += 1
            assert len(self._p_tree) == self.current_proof_depth, "proof_tree must have the same length as current_depth"
            # cancel anything which might got executed
            self._dynamic_proof_executor.cancel_tactic_till_line(tactic_line_num)
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
        else:
            env_info.progress = ProgressState.FAILED
            env_info.error_message = self._dynamic_proof_executor.get_last_exception()
            current_proof_state = state
        return (state, current_proof_state, reward, done, env_info)

    def _get_thms(self, history_idx: int = None):
        assert self._loaded, "Env not loaded, call reset() first"
        history_idx = len(self._history) - 1 if history_idx is None else history_idx
        state, action, current_proof_state, reward, done, env_info = self._history[history_idx]
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

    def _backtrack(self, history_idx: int = None):
        assert self._loaded, "Env not loaded, call reset() first"
        history_idx = len(self._history) - 1 if history_idx is None else history_idx
        state, action, current_proof_state, reward, done, env_info = self._history[history_idx]
        assert action.action_type == ProofAction.ActionType.BACKTRACK, "Action must be of type BACKTRACK"
        last_tactic_line, last_tactic = self._p_tree.try_remove_last_tactic()
        assert (last_tactic is not None and last_tactic_line is not None) or (last_tactic is None and last_tactic_line is None), "last tactic and last tactic line must be either both None or both not None"
        if last_tactic is not None and last_tactic_line is not None:
            try:
                self._dynamic_proof_executor.cancel_tactic_till_line(last_tactic_line)
            except Exception as e:
                history = self._history # History helps us to restore the state
                self.logger.error("Exception occured while backtracking: {}".format(e))
                self.reset() # To ensure that everything is fine
                # Run all tactics in the history
                self._history = history
                run_tactic_idx = []
                for i in range(history_idx):
                    _, action, _, _, _, _ = history[i]
                    if action.action_type == ProofAction.ActionType.RUN_TACTIC:
                        run_tactic_idx.append(i)
                for i in run_tactic_idx[:-1]:
                    self._run_tactic(i) # Run all tactics except the last one which is being backtracked
            if self._dynamic_proof_executor.is_in_proof_mode():
                env_info.progress = ProgressState.RUNNING
                env_info.error_message = "Backtracked successfully"
                reward = 0.0
            else:
                raise Exception("This should never happen as reset() should always take back the environment to a valid proof state in which the proof mode is on")
        else:
            reward = -1.0
            env_info.progress = ProgressState.FAILED
            env_info.error_message = "Cannot backtrack any further"
        current_proof_state = self.state
        done = self.done
        self._history[history_idx] = (state, action, current_proof_state, reward, done, env_info)

    def _foward_to_lemma_proof(self):
        assert self._loaded, "Env not loaded, call reset() first"
        lemma_found = False
        self._lemma_name_with_stmt = None
        while not self._dynamic_proof_executor.execution_complete and not lemma_found:
            assert not self._dynamic_proof_executor.is_in_proof_mode(), "executor must not be in proof mode"
            _ = list(self._dynamic_proof_executor.run_till_next_lemma_return_exec_stmt())
            if self._dynamic_proof_executor.execution_complete:
                break
            lemma_name = self._dynamic_proof_executor.get_lemma_name_if_running().strip()
            lemma_found = lemma_name.startswith(self.lemma_name)
            if not lemma_found:
                _ = list(self._dynamic_proof_executor.run_to_finish_lemma_return_exec())
                if self._dynamic_proof_executor.execution_complete:
                    break

        if not lemma_found:
            raise Exception(f"Could not find lemma {self.lemma_name}")
        self._lemma_name_with_stmt = self._dynamic_proof_executor.get_lemma_stmt_if_running().strip()
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
        elif action_type == ProofAction.ActionType.BACKTRACK:
            return ProofAction(action_type)
        elif action_type == ProofAction.ActionType.EXIT:
            return ProofAction(action_type)
        else:
            raise Exception(f"Invalid action type {action_type}")
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    logger = logging.getLogger(__name__)
    with ProofEnv("test", proof_exec_callback, 'algb_add_comm', max_proof_depth=10, logger=logger) as env:
        done = env.done
        action = scan_action()
        while action.action_type != ProofAction.ActionType.EXIT and not done:
            state, _, _, reward, done, info = env.step(action)
            env.render()
            if not done:
                action = scan_action()
        pass