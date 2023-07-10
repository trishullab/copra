#!/usr/bin/env python3

import sys
root_dir = f"{__file__.split('src')[0]}"
if root_dir not in sys.path:
    sys.path.append(root_dir)

import logging
import typing
from src.tools.coq_executor import CoqExecutor
from src.tools.training_data_format import Goal, LemmaRefWithScore, LemmaReferences, TrainingDataFormat, TrainingDataCollection
from typing import List

class CoqContextHelper(object):
    max_relevance_score = 0.95
    def __init__(self, search_executor: CoqExecutor, depth : typing.Optional[int] = None, logger: logging.Logger = None) -> None:
        assert search_executor is not None, "Search executor cannot be None"
        assert depth is None or depth >= 0, "Depth should be greater than 0"
        self.search_executor = search_executor
        self.depth = depth if depth is not None else -1
        self.logger = logger if logger is not None else logging.getLogger(__name__)
        search_exec_local_lemmas_discovered_so_far = [lemma.split(':')[0].strip() if ':' in lemma else lemma.split()[0].strip() for lemma in self.search_executor.coq.local_lemmas]
        self.search_exec_local_lemmas_discovered_so_far = set([l for l in search_exec_local_lemmas_discovered_so_far if len(l) > 0])

    def _get_local_lemmas_discovered_so_far_set(self, coq_executor: CoqExecutor):
        local_lemmas_discovered_so_far = [lemma.split(':')[0].strip() if ':' in lemma else lemma.split()[0].strip() for lemma in coq_executor.coq.local_lemmas[:-1]]
        local_lemmas_discovered_so_far = set([l for l in local_lemmas_discovered_so_far if len(l) > 0])
        return local_lemmas_discovered_so_far

    def _search_exact_with_diff(self, coq_executor: CoqExecutor, tok: str):
        exact_search_res = self.search_executor.search_exact(tok)
        local_lemmas_discovered_so_far = self._get_local_lemmas_discovered_so_far_set(coq_executor)
        diff = self.search_exec_local_lemmas_discovered_so_far.difference(local_lemmas_discovered_so_far)
        # Remove future lemmas which are yet to be discovered
        exact_search_res = [res for res in exact_search_res if res[0] not in diff]
        return exact_search_res

    def _search_defn_with_diff(self, coq_executor: CoqExecutor, name: str, match_until: typing.Tuple[str, ...], max_search_res: typing.Optional[int] = None):
        search_defn_res = self.search_executor.search_defn(name, match_until, max_search_res=max_search_res)
        local_lemmas_discovered_so_far = self._get_local_lemmas_discovered_so_far_set(coq_executor)
        diff = self.search_exec_local_lemmas_discovered_so_far.difference(local_lemmas_discovered_so_far)
        # Remove future lemmas which are yet to be discovered
        search_defn_res = [res for res in search_defn_res if res[0] not in diff]
        return search_defn_res

    def _get_all_type_matching_defns_with_diff(self, coq_executor: CoqExecutor, tok: str):
        exact_search_res = self.search_executor.get_all_type_matching_defns(tok)
        local_lemmas_discovered_so_far = self._get_local_lemmas_discovered_so_far_set(coq_executor)
        diff = self.search_exec_local_lemmas_discovered_so_far.difference(local_lemmas_discovered_so_far)
        # Remove future lemmas which are yet to be discovered
        exact_search_res = [res for res in exact_search_res if res[0] not in diff]
        return exact_search_res

    def _get_variables_in_hyp(self, hyps: List[str], coq_exec: CoqExecutor) -> typing.Set[str]:
        variables = set()
        for hyp in hyps:
            hyp = hyp.strip()
            defn = hyp.split(":")
            defn_name = defn[0].strip()
            # tokenize defn_name
            possible_vars = set(coq_exec.get_tokens_in_given_stmt(defn_name, ignore_first_token=False))
            possible_vars = {var for var in possible_vars if var not in coq_exec.token_separator_set}
            variables.update(possible_vars)
        return variables

    def _get_changed_goal_idx(self, training_data_point: TrainingDataFormat) -> typing.List[int]:
        # Figure out the subset of start goals which were changed
        start_goals = dict()
        for goal in training_data_point.start_goals:
            if goal.goal in start_goals:
                start_goals[goal.goal] += 1
            else:
                start_goals[goal.goal] = 1
        end_goals = dict()
        for goal in training_data_point.end_goals:
            if goal.goal in end_goals:
                end_goals[goal.goal] += 1
            else:
                end_goals[goal.goal] = 1
        changed_goals = dict()
        for goal, cnt in start_goals.items():
            if goal in end_goals and end_goals[goal] < cnt:
                changed_goals[goal] = cnt - end_goals[goal]
            elif goal not in end_goals:
                changed_goals[goal] = 1
            else:
                # The goal was not changed
                pass
        changed_goals_idx = []
        for idx, goal in enumerate(training_data_point.start_goals):
            if goal.goal in changed_goals and changed_goals[goal.goal] > 0:
                changed_goals_idx.append(idx)
                changed_goals[goal.goal] -= 1
        return changed_goals_idx

    def get_current_goals(self, coq_executor: CoqExecutor) -> List[Goal]:
        # Only consider the foreground goals because we can handle the multi-line tactics
        return [Goal(hypotheses=obligation.hypotheses, goal=obligation.goal) for obligation in coq_executor.coq.proof_context.fg_goals]

    def get_local_lemmas(self, coq_executor: CoqExecutor, logger: logging.Logger = None) -> List[typing.Tuple[str, str]]:
        # Since LOCAL retrieval is not intelligent enough to filter out the useless lemmas, we will not filter out the useless lemmas here, and let the model learn it
        logger = logger if logger is not None else self.logger
        lemmas = []
        for lemma in coq_executor.coq.local_lemmas[:-1]:
            lemma_split = lemma.split(':') if ':' in lemma else lemma.split()
            lemma_name = ""
            if len(lemma_split) > 0:
                lemma_name = lemma_split[0].strip()
            lemma_val = lemma_split[-1].strip()
            if lemma_val.endswith('.'):
                lemma_val = lemma_val[:-1]
            if len(lemma_name) > 0:
                lemmas.append((lemma_name, lemma_val))
        return lemmas

    def set_relevant_defns_in_training_data_point(self, training_data_point: TrainingDataFormat, coq_executor: CoqExecutor, logger: logging.Logger = None, depth: int = None):
        logger = logger if logger is not None else self.logger
        depth = self.depth if depth is None else depth
        unique_defns = {defn: idx for idx, defn in enumerate(training_data_point.all_useful_defns_theorems)}
        for idx, goal in enumerate(training_data_point.start_goals):
            current_depth = 0
            query = training_data_point.get_human_readable_serialized_goal(idx, skip_special_tokens=True)
            possible_variables = self._get_variables_in_hyp(goal.hypotheses, coq_executor)
            query_toks = set(coq_executor.get_tokens_in_given_stmt(query, ignore_first_token=False))
            stack = [(current_depth, tok) for tok in query_toks if tok not in possible_variables]
            toks_executed = set()
            depth_map = {}
            useful_defns_maps : typing.Dict[str, str] = {}
            full_depth = depth == -1
            while len(stack) > 0:
                current_depth, tok = stack.pop()
                if current_depth > depth and not full_depth:
                    continue
                if tok in toks_executed:
                    continue
                else:
                    toks_executed.add(tok)
                for defn, denf_val in self._search_exact_with_diff(coq_executor, tok):
                    if defn in depth_map:
                        depth_map[defn] = min(depth_map[defn], current_depth)
                        useful_defns_maps[defn] = denf_val
                    else:
                        depth_map[defn] = current_depth
                        useful_defns_maps[defn] = denf_val
                    for stmt_tok in coq_executor.get_tokens_in_given_stmt(denf_val, ignore_first_token=False):
                        if stmt_tok not in toks_executed:
                            stack.append((current_depth + 1, stmt_tok))
            useful_defns = [(defn, defn_val, CoqContextHelper.max_relevance_score/(depth_map[defn] + 1)) for defn, defn_val in useful_defns_maps.items()]
            useful_defns.sort(key=lambda x: (x[2], x[0]), reverse=True) # sort by relevance
            for defn, defn_val, _ in useful_defns:
                if defn not in unique_defns:
                    lemma_idx = len(training_data_point.all_useful_defns_theorems)
                    unique_defns[defn] = lemma_idx
                    training_data_point.all_useful_defns_theorems.append(LemmaReferences(lemma_idx, defn, defn_val, 0))
            useful_defns = [LemmaRefWithScore(unique_defns[defn], score) for defn, _, score in useful_defns]
            goal.relevant_defns = useful_defns

    def set_all_type_matched_query_result(self, training_data_point: TrainingDataFormat, coq_executor: CoqExecutor, logger: logging.Logger = None, depth: int = None):
        # Use the hypothesis to find the definition
        # Recursively find the definition of the definition to a fixed depth
        # dump useful_hyps and current stmt into a stack
        logger = logger if logger is not None else self.logger
        depth = self.depth if depth is None else depth
        unique_thms = {defn.lemma_name: idx for idx, defn in enumerate(training_data_point.all_useful_defns_theorems)}
        for idx, goal in enumerate(training_data_point.start_goals):
            current_depth = 0
            query = training_data_point.get_human_readable_serialized_goal(idx, skip_special_tokens=True)
            possible_variables = self._get_variables_in_hyp(goal.hypotheses, coq_executor)
            query_toks = set(coq_executor.get_tokens_in_given_stmt(query, ignore_first_token=False))
            stack = [(current_depth, tok) for tok in query_toks if tok not in possible_variables]
            toks_executed = set()
            depth_map = {}
            useful_defns_maps : typing.Dict[str, str] = {}
            full_depth = depth == -1
            while len(stack) > 0:
                current_depth, tok = stack.pop()
                if current_depth > depth and not full_depth:
                    continue
                if tok in toks_executed:
                    continue
                else:
                    toks_executed.add(tok)
                for defn, denf_val in self._get_all_type_matching_defns_with_diff(coq_executor, tok):
                    if defn in depth_map:
                        depth_map[defn] = min(depth_map[defn], current_depth)
                    else:
                        depth_map[defn] = current_depth
                        useful_defns_maps[defn] = denf_val
                    if current_depth + 1 <= depth or full_depth:
                        for stmt_tok in coq_executor.get_tokens_in_given_stmt(denf_val, ignore_first_token=False):
                            if stmt_tok not in toks_executed:
                                stack.append((current_depth + 1, stmt_tok))
            useful_theorems = [(defn, defn_val, CoqContextHelper.max_relevance_score/(depth_map[defn] + 1)) for defn, defn_val in useful_defns_maps.items()]
            useful_theorems.sort(key=lambda x: (x[2], x[0]), reverse=True) # sort by relevance
            useful_local_theorems = []
            useful_external_theorems = []
            for defn, defn_val, score in useful_theorems:
                if defn not in unique_thms:
                    lemma_idx = len(training_data_point.all_useful_defns_theorems)
                    unique_thms[defn] = lemma_idx
                    training_data_point.all_useful_defns_theorems.append(LemmaReferences(lemma_idx, defn, defn_val, 0))
                if defn in self.search_exec_local_lemmas_discovered_so_far:
                    useful_local_theorems.append((defn, defn_val, score))
                else:
                    useful_external_theorems.append((defn, defn_val, score))
            useful_local_theorems = [(unique_thms[defn], score) for defn, _, score in useful_local_theorems]
            useful_external_theorems = [(unique_thms[defn], score) for defn, _, score in useful_external_theorems]
            goal.used_theorems_local = []
            goal.used_theorems_external = []
            goal.possible_useful_theorems_external = [LemmaRefWithScore(defn_idx, score) for defn_idx, score in useful_external_theorems if score <= CoqContextHelper.max_relevance_score]
            goal.possible_useful_theorems_local = [LemmaRefWithScore(defn_idx, score) for defn_idx, score in useful_local_theorems if score <= CoqContextHelper.max_relevance_score]

    def set_useful_defns_theorems_for_training_data_generation(self, current_stmt: str, training_data_point: TrainingDataFormat, coq_executor: CoqExecutor, logger: logging.Logger = None, depth: int = None, max_search_res: typing.Optional[int] = None):
        # Use the hypothesis to find the definition
        # Recursively find the definition of the definition to a fixed depth
        # dump useful_hyps and current stmt into a stack
        logger = logger if logger is not None else self.logger
        depth = self.depth if depth is None else depth
        current_stmt_toks = tuple(coq_executor.get_tokens_in_given_stmt(current_stmt, ignore_first_token=True)) 
        unique_thms = {defn: idx for idx, defn in enumerate(training_data_point.all_useful_defns_theorems)}
        changed_goal_idx = set(self._get_changed_goal_idx(training_data_point))
        for idx, goal in enumerate(training_data_point.start_goals):
            # if idx not in changed_goal_idx:
            #     continue
            current_depth = 0
            query = training_data_point.get_human_readable_serialized_goal(idx, skip_special_tokens=True)
            possible_variables = self._get_variables_in_hyp(goal.hypotheses, coq_executor)
            query_toks = set(coq_executor.get_tokens_in_given_stmt(query, ignore_first_token=False))
            stack = [(current_depth, tok) for tok in query_toks if tok not in possible_variables]
            toks_executed = set()
            depth_map = {}
            useful_defns_maps : typing.Dict[str, str] = {}
            full_depth = depth == -1
            while len(stack) > 0:
                current_depth, tok = stack.pop()
                if current_depth > depth and not full_depth:
                    continue
                if tok in toks_executed:
                    continue
                else:
                    toks_executed.add(tok)
                for defn, denf_val, is_used in self._search_defn_with_diff(coq_executor, tok, current_stmt_toks, max_search_res=max_search_res):
                    if defn in depth_map:
                        depth_map[defn] = min(depth_map[defn], current_depth)
                        old_val, was_used_earlier = useful_defns_maps[defn]
                        current_defn_val = None
                        if was_used_earlier:
                            current_defn_val = old_val
                        else:
                            current_defn_val = denf_val
                        useful_defns_maps[defn] = (current_defn_val, was_used_earlier or is_used)
                    else:
                        depth_map[defn] = current_depth
                        useful_defns_maps[defn] = (denf_val, is_used)
                    if current_depth + 1 <= depth or full_depth:
                        for stmt_tok in coq_executor.get_tokens_in_given_stmt(denf_val, ignore_first_token=False):
                            if stmt_tok not in toks_executed:
                                stack.append((current_depth + 1, stmt_tok))
            useful_theorems = [(defn, defn_val, 1.0 if is_used else CoqContextHelper.max_relevance_score/(depth_map[defn] + 1)) for defn, (defn_val, is_used) in useful_defns_maps.items()]
            useful_theorems.sort(key=lambda x: (x[2], x[0]), reverse=True) # sort by relevance
            useful_local_theorems = []
            useful_external_theorems = []
            for defn, defn_val, defn_score in useful_theorems:
                if defn not in unique_thms:
                    lemma_idx = len(training_data_point.all_useful_defns_theorems)
                    unique_thms[defn] = lemma_idx
                    training_data_point.all_useful_defns_theorems.append(LemmaReferences(lemma_idx, defn, defn_val, 0)) 
                if defn in self.search_exec_local_lemmas_discovered_so_far:
                    useful_local_theorems.append((defn, defn_val, defn_score))
                else:
                    useful_external_theorems.append((defn, defn_val, defn_score))
            useful_local_theorems = [(unique_thms[defn], defn_score) for defn, _, defn_score in useful_local_theorems]
            useful_external_theorems = [(unique_thms[defn], defn_score) for defn, _, defn_score in useful_external_theorems]
            if idx in changed_goal_idx:
                # Only assign used theorems if it was in changed goals
                goal.used_theorems_local = [LemmaRefWithScore(defn_idx, score) for defn_idx, score in useful_local_theorems if score > CoqContextHelper.max_relevance_score]
                goal.used_theorems_external = [LemmaRefWithScore(defn_idx, score) for defn_idx, score in useful_external_theorems if score > CoqContextHelper.max_relevance_score]
            if len(goal.used_theorems_local) > 0:
                for lemma_ref in goal.used_theorems_local:
                    training_data_point.all_useful_defns_theorems[lemma_ref.lemma_idx].ref_count += 1
            if len(goal.used_theorems_external) > 0:
                for lemma_ref in goal.used_theorems_external:
                    training_data_point.all_useful_defns_theorems[lemma_ref.lemma_idx].ref_count += 1
            goal.possible_useful_theorems_local = [LemmaRefWithScore(defn_idx, score) for defn_idx, score in useful_local_theorems if score <= CoqContextHelper.max_relevance_score]
            goal.possible_useful_theorems_external = [LemmaRefWithScore(defn_idx, score) for defn_idx, score in useful_external_theorems if score <= CoqContextHelper.max_relevance_score]

    def set_local_thms_dfns(self, training_data_point: TrainingDataFormat, coq_executor: CoqExecutor, logger: logging.Logger = None):
        local_lemmas = self.get_local_lemmas(coq_executor, logger)
        unique_thms = {defn.lemma_name: idx for idx, defn in enumerate(training_data_point.all_useful_defns_theorems)}
        useful_local_theorems = []
        for defn, defn_val in local_lemmas:
            if defn not in unique_thms:
                lemma_idx = len(training_data_point.all_useful_defns_theorems)
                unique_thms[defn] = lemma_idx
                training_data_point.all_useful_defns_theorems.append(LemmaReferences(lemma_idx, defn, defn_val, 0))
            if defn in self.search_exec_local_lemmas_discovered_so_far:
                useful_local_theorems.append((defn, defn_val, CoqContextHelper.max_relevance_score))
        for goal in training_data_point.start_goals:
            goal.possible_useful_theorems_local = [LemmaRefWithScore(defn_idx, score) for defn_idx, score in useful_local_theorems if score <= CoqContextHelper.max_relevance_score]

if __name__ == "__main__":
    import os
    project_dir = "data/custom_group_theory/theories"
    file_name = "data/custom_group_theory/theories/grpthm.v"
    coq_exec = CoqExecutor(project_dir, file_name)
    training_data_points = TrainingDataCollection()
    inference_data_points = TrainingDataCollection()
    with CoqExecutor(project_dir, file_name, use_human_readable_proof_context=True) as search_exec:
        with CoqExecutor(project_dir, file_name, use_human_readable_proof_context=True) as coq_exec:
            search_exec.run_to_finish()
            coq_context_helper = CoqContextHelper(search_exec, depth=0)
            # execute coq_exec line by line
            while not coq_exec.execution_complete:
                ran_success, lemma_name = coq_exec.run_till_next_lemma()
                if ran_success:
                    ran_next = True
                    prev_goals : typing.List[Goal] = []
                    while ran_next and coq_exec.is_in_proof_mode():
                        ran_next = coq_exec.run_next()
                        if coq_exec.is_in_proof_mode():
                            training_data_point = TrainingDataFormat()
                            training_data_point.start_goals = [Goal(goal.hypotheses, goal.goal) for goal in prev_goals]
                            training_data_point.end_goals = coq_context_helper.get_current_goals(coq_exec)
                            inference_data_point = TrainingDataFormat()
                            inference_data_point.start_goals = [Goal(goal.hypotheses, goal.goal) for goal in prev_goals]
                            inference_data_point.end_goals = training_data_point.end_goals
                            prev_goals = training_data_point.end_goals
                            coq_context_helper.set_relevant_defns_in_training_data_point(training_data_point, coq_exec)
                            coq_context_helper.set_useful_defns_theorems_for_training_data_generation(coq_exec.current_stmt, training_data_point, coq_exec)
                            training_data_point.proof_steps = [coq_exec.current_stmt]
                            training_data_points.merge(training_data_point)
                            coq_context_helper.set_all_type_matched_query_result(inference_data_point, coq_exec)
                            inference_data_points.merge(inference_data_point)
                            inference_data_point.proof_steps = [coq_exec.current_stmt]
    
    output_path = "data/generated_coq_full_context_dump_very_small_test/"
    os.makedirs(output_path, exist_ok=True)
    training_data_full_serialized = training_data_points.to_json()
    with open(os.path.join(output_path, "training_data_full.json"), "w") as f:
        f.write(training_data_full_serialized)
    inference_data_collection = TrainingDataCollection(inference_data_points)
    inference_data_full_serialized = inference_data_points.to_json()
    with open(os.path.join(output_path, "inference_data_full.json"), "w") as f:
        f.write(inference_data_full_serialized)