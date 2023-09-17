#!/usr/bin/env python3

import sys

root_dir = f"{__file__.split('src')[0]}"
if root_dir not in sys.path:
    sys.path.append(root_dir)
from src.tools.dynamic_coq_proof_exec import DynamicProofExecutor
from src.rl.abstraction import State
from src.tools.training_data_format import TrainingDataFormat
from dataclasses_json import dataclass_json
from dataclasses import dataclass

@dataclass_json
@dataclass
class ProofState(State):
    training_data_format: TrainingDataFormat
    was_reset: bool = False

    def _post_init(self):
        self.proof_tree = None

    @property
    def name(self):
        return f"{self.training_data_format.start_goals}"[:25]

    def __eq__(self, other):
        if not isinstance(other, ProofState):
            return False
        if self.training_data_format is None or other.training_data_format is None:
            return self.training_data_format == other.training_data_format
        desc_cmp = DynamicProofExecutor.goal_description_compare(self.training_data_format.goal_description, other.training_data_format.goal_description)
        if desc_cmp == 0:
            return self.training_data_format == other.training_data_format
        else:
            return False

    def __hash__(self):
        return hash(self.training_data_format)

    def serialize(self) -> str:
        serialized_json = self.to_json()
        return serialized_json
    
    def __ge__(self, __o: object) -> bool:
        assert isinstance(__o, ProofState)
        if __o == FailedProofState: # FailedProofState is the hardest state to reach
            return self.training_data_format == __o.training_data_format
        if self == FailedProofState:
            return True
        assert isinstance(self.training_data_format, TrainingDataFormat)
        desc_cmp = DynamicProofExecutor.goal_description_compare(self.training_data_format.goal_description, __o.training_data_format.goal_description)
        if desc_cmp == 0:
            return self.training_data_format >= __o.training_data_format
        else:
            return desc_cmp < 0

    def __le__(self, __o: object) -> bool:
        assert isinstance(__o, ProofState)
        if self == FailedProofState: # FailedProofState is the hardest state to reach
            return self.training_data_format == __o.training_data_format
        if __o == FailedProofState:
            return True
        assert isinstance(self.training_data_format, TrainingDataFormat)
        desc_cmp = DynamicProofExecutor.goal_description_compare(self.training_data_format.goal_description, __o.training_data_format.goal_description)
        if desc_cmp == 0:
            return self.training_data_format <= __o.training_data_format
        else:
            return desc_cmp > 0
    
    def __lt__(self, __o: object) -> bool:
        assert isinstance(__o, ProofState)
        assert isinstance(self.training_data_format, TrainingDataFormat)
        return self.training_data_format != __o.training_data_format and self.training_data_format <= __o.training_data_format
    
    def __gt__(self, __o: object) -> bool:
        assert isinstance(__o, ProofState)
        assert isinstance(self.training_data_format, TrainingDataFormat)
        return self.training_data_format != __o.training_data_format and self.training_data_format >= __o.training_data_format


FailedProofState = ProofState(training_data_format=None)

if __name__ == "__main__":
    proof_state = ProofState(training_data_format=None)
    print(proof_state.serialize())
    pass