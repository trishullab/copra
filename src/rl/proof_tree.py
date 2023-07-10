#!/usr/bin/env python3

import sys
root_dir = f"{__file__.split('src')[0]}"
if root_dir not in sys.path:
    sys.path.append(root_dir)
import typing
from dataclasses_json import dataclass_json
from dataclasses import dataclass, field
from src.tools.training_data_format import TrainingDataFormat

@dataclass_json
@dataclass
class ProofTree(object):
    tactics: typing.List[typing.Tuple[int, TrainingDataFormat]] = field(default_factory=list)

    def __len__(self):
        return len(self.tactics)

    def __getitem__(self, index):
        return self.tactics[index]

    def try_add_tactic(self, line_num, tactic: TrainingDataFormat, force_add: bool = False):
        # Make sure that the tactic is not more hard than any of the previous tactics
        if not force_add:
            for _, prev_tactic in self.tactics:
                if tactic >= prev_tactic: # New tactic should be easier than all the previous tactics
                    return False
        self.tactics.append((line_num, tactic))
        return True

    def try_remove_last_tactic(self):
        if len(self.tactics) > 0:
            line_num, tactic = self.tactics.pop()
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