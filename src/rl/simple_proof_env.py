#!/usr/bin/env python3

import sys

root_dir = f"{__file__.split('src')[0]}"
if root_dir not in sys.path:
    sys.path.append(root_dir)
import copy
import typing
import logging
import time
import os
from src.rl.proof_tree import ProofSearchResult, ProofTree
from src.rl.proof_state import ProofState
from src.rl.proof_action import ProofAction
from src.rl.abstraction import State, Action, Env
from src.tools.proof_exec_callback import ProofExecutorCallback
from src.tools.training_data_format import TrainingDataFormat
from src.tools.dynamic_lean_proof_exec import DynamicProofExecutor as DynamicLeanProofExecutor
from src.tools.dynamic_coq_proof_exec import DynamicProofExecutor as DynamicCoqProofExecutor
from src.retrieval.coq_bm25_reranker import CoqBm25ReRanker
from src.retrieval.lean3_bm25_reranker import Lean3Bm25ReRanker
from dataclasses import dataclass, field
from dataclasses_json import dataclass_json
from enum import Enum


class ProgressState:
    STARTING = "Starting"
    STATE_CHANGED = "StateChanged"
    STATE_UNCHANGED = "StateUnchanged"
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

    def __eq__(self, __value: object) -> bool:
        return isinstance(__value, ProofEnvInfo) and self.progress == __value.progress and self.error_message == __value.error_message    

class ProofEnvReRankStrategy(Enum):
    BM25 = "BM25"
    BM25_WITH_PRINT = "BM25_WITH_PRINT"
    BM25_WITH_PRINT_ONLY_LOCAL = "BM25_WITH_PRINT_ONLY_LOCAL"
    BM25_WITH_PRINT_NO_DFNS = "BM25_WITH_PRINT_NO_DFNS"
    BM25_WITH_PRINT_ONLY_LOCAL_NO_DFNS = "BM25_WITH_PRINT_ONLY_LOCAL_NO_DFNS"
    BM25_ONLY_LOCAL_NO_DFNS = "BM25_ONLY_LOCAL_NO_DFNS"
    BM25_NO_DFNS = "BM25_NO_DFNS"

    def __str__(self):
        return self.value

class ProofEnv(Env):
    max_depth_penalty = -0.1
    max_proof_completion_reward = 1.0
    progress_reward = 0.2
    _re_ranker = None
    def __init__(self, 
        name: str, 
        dynamic_proof_executor_callback: ProofExecutorCallback,
        lemma_name: str,
        retrieval_strategy: ProofEnvReRankStrategy = ProofEnvReRankStrategy.BM25,
        max_proof_depth: int = 10,
        always_retrieve_thms: bool = False,
        logger : logging.Logger = None):
        assert isinstance(dynamic_proof_executor_callback, ProofExecutorCallback)
        assert isinstance(lemma_name, str)
        assert isinstance(max_proof_depth, int)
        assert isinstance(always_retrieve_thms, bool)
        self.dynamic_proof_executor_callback = dynamic_proof_executor_callback
        self._dynamic_proof_executor : typing.Union[DynamicCoqProofExecutor, DynamicLeanProofExecutor] = None
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
        self._always_retrieve_thms = always_retrieve_thms
        self.retrieve_strategy = retrieval_strategy
        self.language = self.dynamic_proof_executor_callback.language
        if self.retrieve_strategy == ProofEnvReRankStrategy.BM25 or \
            self.retrieve_strategy == ProofEnvReRankStrategy.BM25_WITH_PRINT or \
            self.retrieve_strategy == ProofEnvReRankStrategy.BM25_WITH_PRINT_ONLY_LOCAL or \
            self.retrieve_strategy == ProofEnvReRankStrategy.BM25_WITH_PRINT_NO_DFNS or \
            self.retrieve_strategy == ProofEnvReRankStrategy.BM25_WITH_PRINT_ONLY_LOCAL_NO_DFNS or \
            self.retrieve_strategy == ProofEnvReRankStrategy.BM25_ONLY_LOCAL_NO_DFNS or \
            self.retrieve_strategy == ProofEnvReRankStrategy.BM25_NO_DFNS:
            if ProofEnv._re_ranker is None or str(self.language) != ProofEnv._re_ranker.language:
                if self.language == ProofAction.Language.COQ:
                    ProofEnv._re_ranker = CoqBm25ReRanker(language=str(self.language))
                elif self.language == ProofAction.Language.LEAN:
                    ProofEnv._re_ranker = Lean3Bm25ReRanker(language=str(self.language))
                else:
                    raise NotImplementedError(f"Language {self.language} not implemented")
            self._re_ranker = ProofEnv._re_ranker
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
        use_fallback = True
        if len(self._history) > 0:
            # Just check the last action in history to the current state
            _, _, s2, _, _, _ = self._history[-1]
            if s2 is not None:
                # s2 can be None when called internally for getting the current state before executing an action
                # We need this for actions which keep the state same but add more information like useful theorems and defintions
                current_goals = s2.training_data_format
                use_fallback = False
        if use_fallback:
            # This gets the state from the Coq interface itself
            if self._always_retrieve_thms:
                proof_state, _, _, _ = self._get_current_dfns_thms(ProofEnvInfo(progress=ProgressState.STARTING))
                current_goals = proof_state.training_data_format
            else:
                current_goals = self._dynamic_proof_executor.get_current_proof_state_as_training_data()
        current_goals = copy.deepcopy(current_goals)
        current_proof_tree = copy.deepcopy(self._p_tree)
        lemma_stmt = self._dynamic_proof_executor.get_lemma_stmt_if_running()
        lemma_name = self._dynamic_proof_executor.get_current_lemma_name()
        state = ProofState(current_goals, language=self.language, theorem_statement_with_name=lemma_stmt, theorem_name=lemma_name) # always make a copy of goals to avoid side effects
        state.proof_tree = current_proof_tree
        state.was_reset = len(self._history) == 0
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
            try:
                self._dynamic_proof_executor.__exit__(None, None, None)
            except Exception:
                pass
        self._dynamic_proof_executor = self.dynamic_proof_executor_callback.get_proof_executor()
        if self.dynamic_proof_executor_callback.language == ProofAction.Language.LEAN:
            lean_proof_executor = self._dynamic_proof_executor
            # Initialize the lemma search
            if self._always_retrieve_thms and \
            str(self.language) == str(self.dynamic_proof_executor_callback.language) and \
            len(self._re_ranker.responses) == 0: # This is done only once
                search_tool = lean_proof_executor.lean_context_helper.search_executor._search_tool
                if len(search_tool.lemmas) > 0:
                    all_lemmas = [str(lemma) for lemma in search_tool.lemmas]
                    self._re_ranker.reindex(all_lemmas)
        # if isinstance(self._dynamic_proof_executor, DynamicLeanProofExecutor):
        #     self._always_retrieve_thms = False # Lean does not support retrieval of theorems as of now
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
        elif action.action_type == ProofAction.ActionType.GET_DFNS_THMS:
            self._get_dfns_thms(history_idx)
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

    def dump_proof(self, dump_file_name: str = None, additional_info: typing.Dict[str, typing.Any] = None):
        assert self._loaded, "Env not loaded, call reset() first"
        self.goal_end_time = time.time()
        self.time_taken = self.goal_end_time - self.goal_start_time
        proof_steps = [TrainingDataFormat(proof_steps=tactic.proof_steps) for _, tactic in self._p_tree.tactics]
        additional_info = additional_info if additional_info is not None else {}
        self.proof_search_res = ProofSearchResult(
            self._dynamic_proof_executor.main_file, 
            not self._dynamic_proof_executor.is_in_proof_mode(), 
            self._lemma_name_with_stmt, 
            proof_steps, 
            self.time_taken, 
            self.inferences_used, 
            possible_failed_paths=-1, 
            num_of_backtracks=-1, 
            is_timeout=False, 
            is_inference_exhausted=False, 
            longest_success_path=-1,
            additional_info=additional_info,
            language=self.language)
        self.logger.info(f"Dumping proof search result:\n {self.proof_search_res}")
        if dump_file_name is not None:
            opening_mode = 'a' if os.path.exists(dump_file_name) else 'w'
            with open(dump_file_name, opening_mode) as f:
                if opening_mode == 'a':
                    f.write("\n\n")
                f.write(str(self.proof_search_res))

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
        try:
            state, next_state, reward, done, env_info = self._run_tactics(tactics, state, action, env_info)
        except Exception:
            self.logger.exception(f"Exception occured while running tactics:\n {tactics}")
            self.logger.info("Resetting the environment and running all the tactics again")
            self._reset_and_restore_history()
            next_state = self.state
            reward = -1.0
            done = False
            env_info.progress = ProgressState.FAILED
            env_info.error_message = self._dynamic_proof_executor.get_last_exception()
        self._history[history_idx] = (state, action, next_state, reward, done, env_info)

    def _run_tactics(self, tactics: typing.List[str], state: ProofState, action: ProofAction, env_info: ProofEnvInfo):
        env_info = copy.deepcopy(env_info)
        tactic_line_num, ran_successfully = self._dynamic_proof_executor.run_tactics(tactics)
        proof_progressed = False
        if ran_successfully:
            previous_proof_state = state
            previous_proof_state.training_data_format.proof_steps = copy.deepcopy(tactics)
            # add the proof step to the proof tree
            self._p_tree.try_add_tactic(tactic_line_num, previous_proof_state.training_data_format, force_add=True, action=action)
            self.current_proof_depth += 1
            proof_progressed = True
            current_proof_state = self.state
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
            env_info.progress = ProgressState.STATE_CHANGED if state != current_proof_state else ProgressState.STATE_UNCHANGED
            env_info.error_message = None
        else:
            env_info.progress = ProgressState.FAILED
            env_info.error_message = self._dynamic_proof_executor.get_last_exception()
            current_proof_state = copy.deepcopy(state)
            # There is a special case of the first tactic failing, in which case there is no reset
            # So always decide the reset based on whether the history is empty or not
            # Clone the current_proof_state always to avoid side effects
            current_proof_state.was_reset = len(self._history) == 0
        return (state, current_proof_state, reward, done, env_info)

    def _get_dfns_thms(self, history_idx: int = None):
        assert self._loaded, "Env not loaded, call reset() first"
        history_idx = len(self._history) - 1 if history_idx is None else history_idx
        state, action, current_proof_state, reward, done, env_info = self._history[history_idx]
        assert action.action_type == ProofAction.ActionType.GET_DFNS_THMS, "Action must be of type GET_DFNS_THMS"
        current_proof_state, reward, done, env_info = self._get_current_dfns_thms(env_info)
        self._history[history_idx] = (state, action, current_proof_state, reward, done, env_info)

    def _get_current_dfns_thms(self, env_info : ProofEnvInfo):
        should_print_symbol = self.retrieve_strategy == ProofEnvReRankStrategy.BM25_WITH_PRINT or \
        self.retrieve_strategy == ProofEnvReRankStrategy.BM25_WITH_PRINT_ONLY_LOCAL or \
        self.retrieve_strategy == ProofEnvReRankStrategy.BM25_WITH_PRINT_NO_DFNS or \
        self.retrieve_strategy == ProofEnvReRankStrategy.BM25_WITH_PRINT_ONLY_LOCAL_NO_DFNS
        should_have_relevant_dfns = self.retrieve_strategy == ProofEnvReRankStrategy.BM25_WITH_PRINT or \
        self.retrieve_strategy == ProofEnvReRankStrategy.BM25_WITH_PRINT_ONLY_LOCAL or \
        self.retrieve_strategy == ProofEnvReRankStrategy.BM25
        only_local = self.retrieve_strategy == ProofEnvReRankStrategy.BM25_WITH_PRINT_ONLY_LOCAL or \
        self.retrieve_strategy == ProofEnvReRankStrategy.BM25_WITH_PRINT_ONLY_LOCAL_NO_DFNS or \
        self.retrieve_strategy == ProofEnvReRankStrategy.BM25_ONLY_LOCAL_NO_DFNS
        relevant_defns_thms = self._dynamic_proof_executor.get_all_relevant_defns_and_thms(should_print_symbol, only_local)
        if should_have_relevant_dfns:
            for idx, goal in enumerate(relevant_defns_thms.start_goals):
                query = relevant_defns_thms.get_human_readable_serialized_goal(idx, skip_special_tokens=True)
                responses = [str(relevant_defns_thms.all_useful_defns_theorems[lemma_ref.lemma_idx]) for lemma_ref in goal.relevant_defns]
                if len(self._re_ranker.responses) > 0 and len(responses) == len(self._re_ranker.responses):
                    response_scores = self._re_ranker.get_scores(query) # When the response are globally same
                else:
                    response_scores = self._re_ranker.rerank(query, responses)
                relevant_defns_idx = [(idx, score) for idx, score in enumerate(response_scores)]
                relevant_defns_idx.sort(key=lambda x: x[1], reverse=True)
                relevant_defns_reranked = [goal.relevant_defns[idx] for idx, _ in relevant_defns_idx]
                sum_scores = sum([score for _, score in relevant_defns_idx]) + 1e-6
                for i in range(len(relevant_defns_reranked)):
                    relevant_defns_reranked[i].score = relevant_defns_idx[i][1]/sum_scores
                goal.relevant_defns = relevant_defns_reranked
        else:
            for goal in relevant_defns_thms.start_goals:
                goal.relevant_defns = []

        for idx, goal in enumerate(relevant_defns_thms.start_goals):
            query = relevant_defns_thms.get_human_readable_serialized_goal(idx, skip_special_tokens=True)
            local_responses = [str(relevant_defns_thms.all_useful_defns_theorems[lemma_ref.lemma_idx]) for lemma_ref in goal.possible_useful_theorems_local]
            if self.retrieve_strategy == ProofEnvReRankStrategy.BM25_WITH_PRINT_ONLY_LOCAL:
                global_responses = []
            else:
                global_responses = [str(relevant_defns_thms.all_useful_defns_theorems[lemma_ref.lemma_idx]) for lemma_ref in goal.possible_useful_theorems_external]
            if len(self._re_ranker.responses) > 0 and len(local_responses) == len(self._re_ranker.responses):
                local_scores = self._re_ranker.get_scores(query)
            else:
                local_scores = self._re_ranker.rerank(query, local_responses)
            if len(self._re_ranker.responses) > 0 and len(global_responses) == len(self._re_ranker.responses):
                global_scores = self._re_ranker.rerank(query, global_responses)
            else:
                global_scores = self._re_ranker.rerank(query, global_responses)
            local_idx = [(idx, score) for idx, score in enumerate(local_scores)]
            global_idx = [(idx, score) for idx, score in enumerate(global_scores)]
            local_idx.sort(key=lambda x: x[1], reverse=True)
            global_idx.sort(key=lambda x: x[1], reverse=True)
            local_responses = [goal.possible_useful_theorems_local[idx] for idx, _ in local_idx]
            global_responses = [goal.possible_useful_theorems_external[idx] for idx, _ in global_idx]
            # Remove any local responses which are already in the relevant defns
            relevant_dfns_names = set([relevant_defns_thms.all_useful_defns_theorems[lemma_ref.lemma_idx].lemma_name for lemma_ref in goal.relevant_defns])
            local_responses = [response for response in local_responses if relevant_defns_thms.all_useful_defns_theorems[response.lemma_idx].lemma_name not in relevant_dfns_names]
            # Remove any global responses which are already in the relevant defns
            global_responses = [response for response in global_responses if relevant_defns_thms.all_useful_defns_theorems[response.lemma_idx].lemma_name not in relevant_dfns_names]
            sum_local_scores = sum([score for _, score in local_idx]) + 1e-6
            sum_global_scores = sum([score for _, score in global_idx]) + 1e-6
            for i in range(len(local_responses)):
                local_responses[i].score = local_idx[i][1]/sum_local_scores
            for i in range(len(global_responses)):
                global_responses[i].score = global_idx[i][1]/sum_global_scores
            goal.possible_useful_theorems_local = local_responses
            goal.possible_useful_theorems_external = global_responses
        lemma_stmt = self._dynamic_proof_executor.get_lemma_stmt_if_running()
        current_proof_state = ProofState(relevant_defns_thms, language=self.language, theorem_statement_with_name=lemma_stmt)
        current_proof_state.proof_tree = copy.deepcopy(self._p_tree)
        done = self.done
        env_info.progress = ProgressState.STATE_UNCHANGED if not done else ProgressState.DONE
        env_info.error_message = None
        reward = 0.0
        return current_proof_state, reward, done, env_info

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
                self.current_proof_depth -= 1
            except Exception:
                # history = self._history # History helps us to restore the state
                self.logger.exception("Exception occured while backtracking")
                history = copy.deepcopy(self._history)
                p_tree = copy.deepcopy(self._p_tree)
                self.reset() # To ensure that everything is fine we start again
                # Run all the current steps in the proof tree
                self.logger
                for _, tactic in p_tree.tactics:
                    self._run_tactics(tactic.proof_steps, self.state, ProofEnvInfo(progress=ProgressState.STARTING))
                    # No need to capture in history as the history is already captured
                self._history = history
                self.logger.warning("Backtracking failed, resetting the environment and running all the tactics again till two-steps before the backtracked step (hence effectively backtracking!)")

            if self._dynamic_proof_executor.is_in_proof_mode():
                env_info.progress = ProgressState.STATE_CHANGED
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
        if isinstance(self._dynamic_proof_executor, DynamicCoqProofExecutor):
            while not self._dynamic_proof_executor.execution_complete and not lemma_found:
                assert not self._dynamic_proof_executor.is_in_proof_mode(), "executor must not be in proof mode"
                _ = list(self._dynamic_proof_executor.run_till_next_lemma_return_exec_stmt())
                if self._dynamic_proof_executor.execution_complete:
                    break
                lemma_name = self._dynamic_proof_executor.get_lemma_name_if_running()
                if lemma_name is not None:
                    lemma_name = lemma_name.strip()
                lemma_found = lemma_name.startswith(self.lemma_name) if lemma_name is not None else False
                if not lemma_found:
                    _ = list(self._dynamic_proof_executor.run_to_finish_lemma_return_exec())
                    if self._dynamic_proof_executor.execution_complete:
                        break
        elif isinstance(self._dynamic_proof_executor, DynamicLeanProofExecutor):
            self._dynamic_proof_executor.skip_to_theorem(self.lemma_name)
            lemma_found = True
        else:
            raise NotImplementedError(f"Proof executor {type(self._dynamic_proof_executor)} not implemented")

        if not lemma_found:
            raise Exception(f"Could not find lemma {self.lemma_name}")
        self._lemma_name_with_stmt = self._dynamic_proof_executor.get_lemma_stmt_if_running().strip()
        pass

    def _reset_and_restore_history(self):
        history = copy.deepcopy(self._history)
        p_tree = copy.deepcopy(self._p_tree)
        self.reset() # To ensure that everything is fine we start again
        # Run all the current steps in the proof tree
        self.logger
        for (_, tactic), action in zip(p_tree.tactics, p_tree.actions):
            self._run_tactics(tactic.proof_steps, self.state, action, ProofEnvInfo(progress=ProgressState.STARTING))
            # No need to capture in history as the history is already captured
        self._history = history


if __name__ == "__main__":
    import os
    os.chdir(root_dir)

    print("Interactive Proof Environment")
    supported_actions = [x.name for x in ProofAction.ActionType]

    def scan_action(language):
        inp_action_type = input(f"Enter an action type from {supported_actions}: (default RUN_TACTIC)")
        if inp_action_type not in supported_actions:
            inp_action_type = ProofAction.ActionType.RUN_TACTIC.name
        action_type = ProofAction.ActionType[inp_action_type]
        if action_type == ProofAction.ActionType.RUN_TACTIC:
            inp = input("Enter tactic(s) (';' separated): ")
            inp = inp.split(';')
            return ProofAction(action_type, language, tactics=inp)
        elif action_type == ProofAction.ActionType.GET_DFNS_THMS or action_type == ProofAction.ActionType.BACKTRACK or action_type == ProofAction.ActionType.EXIT:
            return ProofAction(action_type, language)
        else:
            raise Exception(f"Invalid action type {action_type}")
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    inp = input("Want to run coq or lean env? (Enter 'coq'/'lean') ")
    language = ProofAction.Language.COQ
    if inp == 'coq':
        proof_exec_callback = ProofExecutorCallback(
            project_folder=".",
            file_path="data/test/SimpleAlgebra.v"
        )
        theorem_name = "algb_add_comm"
        language = ProofAction.Language.COQ
        always_retrieve_thms = False
    elif inp == 'lean':
        proof_exec_callback = ProofExecutorCallback(
            project_folder="data/benchmarks/miniF2F",
            file_path="data/benchmarks/miniF2F/lean/src/test.lean",
            language=ProofAction.Language.LEAN,
            always_use_retrieval=True
        )
        theorem_name = "mathd_algebra_478"
        language = ProofAction.Language.LEAN
        always_retrieve_thms = True
        pass
    else:
        raise Exception(f"Invalid input {inp} for choosing coq/lean")
    logger = logging.getLogger(__name__)
    with ProofEnv("test", proof_exec_callback, theorem_name, max_proof_depth=10, always_retrieve_thms=always_retrieve_thms, logger=logger) as env:
        done = env.done
        action = scan_action(language)
        while action.action_type != ProofAction.ActionType.EXIT and not done:
            state, _, _, reward, done, info = env.step(action)
            env.render()
            if not done:
                action = scan_action(language)
        pass