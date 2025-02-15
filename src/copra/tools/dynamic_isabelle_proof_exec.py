#!/usr/bin/env python3

import sys

root_dir = f"{__file__.split('src')[0]}"
if root_dir not in sys.path:
    sys.path.append(root_dir)
import typing
import os
import copy
import enum
import logging
from src.rl.proof_action import ProofAction
from src.tools.training_data_format import Goal, TrainingDataFormat
from src.tools.isabelle_parse_utils import IsabelleLineByLineReader
from src.tools.isabelle_executor import IsabelleExecutor
from src.tools.isabelle_context_helper import IsabelleContextHelper

class IntertwinedIterator(object):
    def __init__(self, iterator: typing.Optional[typing.Iterator[str]] = None):
        self.base_iterator = iterator
        self.next_instruction: typing.Optional[str] = None
        self.base_iterator_stopped = iterator is None # if the base iterator is None, then it is stopped
    
    def set_next_instruction(self, instruction: str):
        assert self.next_instruction is None, "next_instruction must be None"
        assert instruction is not None, "instruction must not be None"
        self.next_instruction = instruction
        
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.next_instruction is not None:
            # Return the next instruction if it is set
            next_instruction = self.next_instruction
            self.next_instruction = None
            return next_instruction
        # Otherwise, get the next instruction from the base iterator
        if self.base_iterator is not None and not self.base_iterator_stopped:
            try:
                instruction = next(self.base_iterator)
                return instruction
            except StopIteration:
                self.base_iterator_stopped = True
                raise
        else:
            raise StopIteration()
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.base_iterator is not None:
            self.base_iterator.close()
        pass

class DynamicProofExecutor(IsabelleExecutor):
    class RunState(object):
        def __init__(self):
            self.tatics_ran = []
            self.last_exception : typing.Optional[str] = None
            self.line_tactic_map = {}
            self.line_proof_context_map = {}
    UnfocussedGoalsDescription = "There are unfocussed goals."
    ProofFinishedDescription = "Proof finished."
    NotInProofModeDescription = "Not in proof mode."
    GoalDescriptionOrder = {
        UnfocussedGoalsDescription: 2, # This more hard
        ProofFinishedDescription: 1, # This is easier coz proof is almost done
        NotInProofModeDescription: 0 # This is the easiest as the proof is done
    }
    class ContextType(enum.Enum):
        NoContext = 0
        LocalContext = 1
        BestContext = 2

    def goal_description_compare(description1: str, descripton2: str) -> int:
        """
        Returns 1 if description1 < description2, 0 if description1 == description2, -1 if description1 > description2
        """
        # In case of no description it is much more harder as we have to do a lot of work
        # So None will have same value as unfocussed goals
        order1 = DynamicProofExecutor.GoalDescriptionOrder.get(description1, 2) if description1 is not None else 2
        order2 = DynamicProofExecutor.GoalDescriptionOrder.get(descripton2, 2) if descripton2 is not None else 2
        if order1 < order2:
            return 1
        elif order1 == order2:
            return 0
        else:
            return -1


    def __init__(self, isabelle_context_helper: IsabelleContextHelper, project_folder: str = None, proof_file: str = None, instruction_iter: typing.Optional[str] = None, use_hammer: ProofAction.HammerMode = ProofAction.HammerMode.AUTO, timeout_in_seconds: int = 60, use_human_readable_proof_context: bool = True, suppress_error_log: bool = True, context_type: ContextType = ContextType.NoContext):
        assert proof_file is None or os.path.exists(proof_file), f"Proof file {proof_file} does not exist"
        assert isabelle_context_helper is not None, "isabelle_context_helper must not be None"
        self.proof_file = proof_file
        self.context_type = context_type
        self.isabelle_file_iter = IsabelleLineByLineReader(proof_file).instruction_step_generator() if proof_file is not None else instruction_iter
        self.tactic_switch_iterator = IntertwinedIterator(self.isabelle_file_iter)
        self.run_state = DynamicProofExecutor.RunState()
        self.logger = None
        self.isabelle_context_helper = isabelle_context_helper
        super().__init__(project_root=project_folder, main_file=proof_file, proof_step_iter=self.tactic_switch_iterator, use_hammer=use_hammer, timeout_in_sec=timeout_in_seconds, use_human_readable_proof_context=use_human_readable_proof_context, suppress_error_log=suppress_error_log)

    def __enter__(self):
        self.isabelle_context_helper.__enter__()
        return super().__enter__()
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.isabelle_context_helper.__exit__(exc_type, exc_val, exc_tb)
        super().__exit__(exc_type, exc_val, exc_tb)

    def set_logger(self, logger: logging.Logger):
        self.logger = logger
        pass

    def get_focussed_goals(self) -> typing.List[Goal]:
        if not self.is_in_proof_mode():
            return []
        return self.isabelle_context_helper.get_focussed_goals(self)
    
    def get_unfocussed_goals(self) -> typing.List[Goal]:
        if not self.is_in_proof_mode():
            return []
        return self.isabelle_context_helper.get_unfocussed_goals(self)

    def get_current_proof_state_as_training_data(self) -> TrainingDataFormat:
        # get the current goal
        if self.needs_cut_close():
            current_goals = self.get_unfocussed_goals()
            training_data_format = TrainingDataFormat(start_goals=current_goals)
            training_data_format.goal_description = DynamicProofExecutor.UnfocussedGoalsDescription
        elif not self.is_in_proof_mode():
            current_goals = self.get_focussed_goals()
            training_data_format = TrainingDataFormat(start_goals=current_goals)
            training_data_format.goal_description = DynamicProofExecutor.NotInProofModeDescription
        elif self.needs_qed():
            current_goals = self.get_focussed_goals()
            assert len(current_goals) == 0, "There should be no goals when needs_qed is True"
            training_data_format = TrainingDataFormat(start_goals=current_goals)
            training_data_format.goal_description = DynamicProofExecutor.ProofFinishedDescription
        else:
            current_goals = self.get_focussed_goals()
            training_data_format = TrainingDataFormat(start_goals=current_goals)
            training_data_format.goal_description = None
        return training_data_format
    
    def get_all_relevant_thms(self, should_print_symbol: bool = False) -> TrainingDataFormat:
        training_data_format = self.get_current_proof_state_as_training_data()
        self.isabelle_context_helper.set_all_type_matched_query_result(training_data_format, self, self.logger)
        return training_data_format
    
    def get_all_relevant_thms_within_local_context(self) -> TrainingDataFormat:
        training_data_format = self.get_current_proof_state_as_training_data()
        self.isabelle_context_helper.set_local_thms_dfns(training_data_format, self, self.logger)
        return training_data_format
    
    def get_all_relevant_defns(self) -> TrainingDataFormat:
        training_data_format = self.get_current_proof_state_as_training_data()
        self.isabelle_context_helper.set_relevant_defns_in_training_data_point(training_data_format, self, self.logger)
        return training_data_format
    
    def get_all_relevant_defns_and_thms(self, should_print_symbol: bool = False, only_local: bool = False) -> TrainingDataFormat:
        training_data_format = self.get_current_proof_state_as_training_data()
        # self.isabelle_context_helper.set_relevant_defns_in_training_data_point(training_data_format, self, self.logger)
        self.isabelle_context_helper.set_all_type_matched_query_result(training_data_format, self, self.logger)
        return training_data_format

    def run_cmds(self, cmds: typing.List[str], raise_exception=False) -> typing.Tuple[int, bool]:
        cmd_failed = False
        start_line_num = self.line_num
        for cmd in cmds:
            self.tactic_switch_iterator.set_next_instruction(cmd)
            try:
                self.run_next()
            except Exception:
                self.line_num -= 1
                cmd_failed = True
                if raise_exception:
                    raise
                else:
                    break
        return start_line_num, not cmd_failed

    def run_tactics(self, tactics: typing.List[str]) -> typing.Tuple[int, bool]:
        tactic_failed = False
        start_line_num = self.line_num
        self.run_state.line_tactic_map[self.line_num] = len(self.run_state.tatics_ran)
        self.run_state.line_proof_context_map[self.line_num] = copy.deepcopy(self.proof_context)
        for idx, tactic in enumerate(tactics):
            self.tactic_switch_iterator.set_next_instruction(tactic)
            try:
                actual_tactic = self.run_next()
                tactics[idx] = actual_tactic
                self.run_state.tatics_ran.append(actual_tactic)
                self.run_state.line_proof_context_map[self.line_num] = copy.deepcopy(self.proof_context)
            except Exception as e:
                self.line_num -= 1
                tactic_failed = True
                self.run_state.last_exception = str(e)
                break
        return start_line_num, not tactic_failed
    
    def get_last_exception(self) -> typing.Optional[str]:
        last_exception = self.run_state.last_exception
        self.run_state.last_exception = None
        return last_exception

    def cancel_tactic_till_line(self, tactic_line_num: int) -> bool:
        assert tactic_line_num <= self.line_num, "tactic_line_num must be <= self.line_num"
        assert tactic_line_num >= 0, "tactic_line_num must be >= 0"
        cancelled_some_tactics = False
        if tactic_line_num < self.line_num:
            state_num = self.run_state.line_tactic_map[tactic_line_num]
            self.run_state.tatics_ran = self.run_state.tatics_ran[:state_num]
            self.proof_context = self.run_state.line_proof_context_map[tactic_line_num]
            line_tactic_map_keys = list(self.run_state.line_tactic_map.keys())
            for line_num in line_tactic_map_keys:
                if line_num >= tactic_line_num:
                    del self.run_state.line_tactic_map[line_num]
            line_proof_context_map_keys = list(self.run_state.line_proof_context_map.keys())
            for line_num in line_proof_context_map_keys:
                if line_num >= tactic_line_num:
                    del self.run_state.line_proof_context_map[line_num]
            self.line_num = tactic_line_num
            assert self.line_num_to_state[self.line_num] is not None, "Nonexistent state found on line " + str(self.line_num)
            self.current_state = self.line_num_to_state[self.line_num]
            cancelled_some_tactics = True
        return cancelled_some_tactics