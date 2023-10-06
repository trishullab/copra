#!/usr/bin/env python3

import sys

root_dir = f"{__file__.split('src')[0]}"
if root_dir not in sys.path:
    sys.path.append(root_dir)
import typing
from dataclasses_json import dataclass_json
from dataclasses import dataclass, field
from src.rl.proof_action import ProofAction
from src.tools.training_data_format import TrainingDataFormat

@dataclass_json
@dataclass
class ProofTree(object):
    tactics: typing.List[typing.Tuple[int, TrainingDataFormat]] = field(default_factory=list)
    actions: typing.List[typing.Optional[ProofAction]] = field(default_factory=list)

    def __len__(self):
        return len(self.tactics)

    def __getitem__(self, index):
        return self.tactics[index]

    def try_add_tactic(self, line_num, tactic: TrainingDataFormat, force_add: bool = False, action: ProofAction = None):
        # Make sure that the tactic is not more hard than any of the previous tactics
        if not force_add:
            for _, prev_tactic in self.tactics:
                if tactic >= prev_tactic: # New tactic should be easier than all the previous tactics
                    return False
        self.tactics.append((line_num, tactic))
        self.actions.append(action)
        return True

    def try_remove_last_tactic(self):
        if len(self.tactics) > 0:
            line_num, tactic = self.tactics.pop()
            self.actions.pop()
            return line_num, tactic
        return None, None

    def _convert_to_str(self, tactic: TrainingDataFormat) -> typing.Tuple[str, list, list]:
        # sort the goals
        goal_set = set([goal.goal for goal in tactic.start_goals])
        hyp_set = set([hyp for goal in tactic.start_goals for hyp in goal.hypotheses])
        goals = sorted(goal_set)
        hyps = sorted(hyp_set)
        goal_str = "\n".join(goals)
        return goal_str, goals, hyps

@dataclass_json
@dataclass
class ProofSearchResult(object):
    proof_file: typing.Optional[str]
    proof_found: bool
    lemma_name: str
    proof_steps: typing.List[TrainingDataFormat]
    proof_time_in_secs: float
    inferences_taken: int
    possible_failed_paths: int
    num_of_backtracks: int
    is_timeout: bool
    is_inference_exhausted: bool
    longest_success_path: int
    additional_info: typing.Dict[str, typing.Any] = field(default_factory=dict)
    language: ProofAction.Language = ProofAction.Language.COQ

    def __str__(self) -> str:
        try:
            lines = [line for step in self.proof_steps for line in step.proof_steps]
            proof_metadata = f"""
ProofFile: {self.proof_file}
LemmaName: {self.lemma_name}
SearchResult: {'[FAILED]' if not self.proof_found else '[SUCCESS]'}
IsInferenceExhausted: {self.is_inference_exhausted}
IsTimeout: {self.is_timeout}
LongestSuccessPath: {self.longest_success_path} 
StepsUsed: {self.inferences_taken}
SearchTimeInSecs: {self.proof_time_in_secs}
NumberOfBacktracks: {self.num_of_backtracks}
PossibleFailedPaths: {self.possible_failed_paths}
AdditionalInfo: {self.additional_info}
"""
            proof_start = "Proof." if self.language == ProofAction.Language.COQ else "begin"
            proof_end = "Qed." if self.language == ProofAction.Language.COQ else "end"
            all_proof_steps = "\n    ".join(lines[:-1]) if len(lines) > 1 else ""
            last_line = (lines[-1] if lines[-1] == proof_end else f"    {lines[-1]}\n") if len(lines) > 0 else ""
            return f"""{self.lemma_name}
{proof_start}
    {all_proof_steps}
{last_line}
{proof_metadata}
"""
        except Exception:
            return f"------- UNABLE_TO_PRINT_INCOMPLETE_PROOF ----------"