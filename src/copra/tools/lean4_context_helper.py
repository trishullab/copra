#!/usr/bin/env python3

import sys
root_dir = f"{__file__.split('src')[0]}"
if root_dir not in sys.path:
    sys.path.append(root_dir)

import logging
import typing
from src.tools.lean4_sync_executor import Lean4SyncExecutor
from src.tools.training_data_format import Goal, LemmaRefWithScore, LemmaReferences, TrainingDataFormat
from typing import List

class Lean4ContextHelper(object):
    max_relevance_score = 0.95
    def __init__(self, search_executor: Lean4SyncExecutor, depth : typing.Optional[int] = None, logger: logging.Logger = None) -> None:
        assert search_executor is not None, "Search executor cannot be None"
        assert depth is None or depth >= 0, "Depth should be greater than 0"
        self.search_executor = search_executor
        self.depth = depth if depth is not None else -1
        self.logger = logger if logger is not None else logging.getLogger(__name__)
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        pass

    def get_focussed_goals(self, lean_executor: Lean4SyncExecutor) -> List[Goal]:
        # Only consider the foreground goals because we can handle the multi-line tactics
        return [Goal(hypotheses=goal.hypotheses, goal=goal.goal) for goal in lean_executor.proof_context.fg_goals]
    
    def get_unfocussed_goals(self, lean_executor: Lean4SyncExecutor) -> List[Goal]:
        # Only consider the foreground goals because we can handle the multi-line tactics
        other_goals = lean_executor.proof_context.bg_goals + lean_executor.proof_context.shelved_goals + lean_executor.proof_context.given_up_goals
        return [Goal(hypotheses=goal.hypotheses, goal=goal.goal) for goal in other_goals]

    def get_local_lemmas(self, lean_executor: Lean4SyncExecutor, logger: logging.Logger = None) -> List[typing.Tuple[str, str]]:
        # Search is not supported in Lean as of now
        raise Exception("Search is not supported in Lean as of now")


    def set_relevant_defns_in_training_data_point(self, training_data_point: TrainingDataFormat, lean_executor: Lean4SyncExecutor, logger: logging.Logger = None, depth: int = None):
        # Search is not supported in Lean as of now
        raise Exception("Search is not supported in Lean as of now")

    def set_all_type_matched_query_result(self, training_data_point: TrainingDataFormat, lean_executor: Lean4SyncExecutor, logger: logging.Logger = None, depth: int = None):
        unique_thms = {defn.lemma_name: idx for idx, defn in enumerate(training_data_point.all_useful_defns_theorems)}
        # query = training_data_point.get_human_readable_serialized_goal(idx, skip_special_tokens=True)
        relevant_thms = self.search_executor.search_type_matching_defns("") # Here the search simply returns everything
        # Add all lemma references to unique_defns
        for thm in relevant_thms:
            thm_name = f"{thm.namespace}.{thm.name}"
            if thm.name not in unique_thms:
                _idx = len(training_data_point.all_useful_defns_theorems)
                unique_thms[thm_name] = _idx
                training_data_point.all_useful_defns_theorems.append(LemmaReferences(_idx, thm_name, thm.dfn))
        for _, goal in enumerate(training_data_point.start_goals):
            goal.possible_useful_theorems_external = [LemmaRefWithScore(unique_thms[f"{thm.namespace}.{thm.name}"], 1.0) for thm in relevant_thms]
            goal.possible_useful_theorems_local = []

    def set_useful_defns_theorems_for_training_data_generation(self, current_stmt: str, training_data_point: TrainingDataFormat, lean_executor: Lean4SyncExecutor, logger: logging.Logger = None, depth: int = None, max_search_res: typing.Optional[int] = None):
        # Search is not supported in Lean as of now
        raise Exception("Search is not supported in Lean as of now")

    def set_local_thms_dfns(self, training_data_point: TrainingDataFormat, lean_executor: Lean4SyncExecutor, logger: logging.Logger = None):
        # Search is not supported in Lean as of now
        raise Exception("Search is not supported in Lean as of now")        
