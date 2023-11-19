#!/usr/bin/env python3

import sys

root_dir = f"{__file__.split('src')[0]}"
if root_dir not in sys.path:
    sys.path.append(root_dir)
import typing
from src.tools.dynamic_coq_proof_exec import DynamicProofExecutor as DynamicCoqProofExecutor
from src.tools.dynamic_lean_proof_exec import DynamicProofExecutor as DynamicLeanProofExecutor
from src.rl.abstraction import State
from src.rl.proof_action import ProofAction
from src.tools.training_data_format import TrainingDataFormat
from dataclasses_json import dataclass_json
from dataclasses import dataclass

@dataclass_json
@dataclass
class ProofState(State):
    training_data_format: TrainingDataFormat
    was_reset: bool = False
    language: ProofAction.Language = ProofAction.Language.COQ
    theorem_statement_with_name: typing.Optional[str] = None
    theorem_name: typing.Optional[str] = None

    def _post_init(self):
        self.proof_tree = None

    @property
    def name(self):
        return f"{self.training_data_format.start_goals}"[:25]

    def __eq__(self, other):
        if not isinstance(other, ProofState):
            return False
        assert self.language == other.language, f"self.language: {self.language}, other.language: {other.language}"
        if self.training_data_format is None or other.training_data_format is None:
            return self.training_data_format == other.training_data_format
        if self.language == ProofAction.Language.COQ:
            desc_cmp = DynamicCoqProofExecutor.goal_description_compare(self.training_data_format.goal_description, other.training_data_format.goal_description)
        elif self.language == ProofAction.Language.LEAN:
            desc_cmp = DynamicLeanProofExecutor.goal_description_compare(self.training_data_format.goal_description, other.training_data_format.goal_description)
        else:
            raise NotImplementedError(f"language {self.language} not supported")
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
        assert self.language == __o.language, f"self.language: {self.language}, __o.language: {__o.language}"
        FailedProofState = FailedCoqProofState if self.language == ProofAction.Language.COQ else FailedLeanProofState
        if __o == FailedProofState: # FailedProofState is the hardest state to reach
            return self.training_data_format == __o.training_data_format
        if self == FailedProofState:
            return True
        assert isinstance(self.training_data_format, TrainingDataFormat)
        if self.language == ProofAction.Language.COQ:
            desc_cmp = DynamicCoqProofExecutor.goal_description_compare(self.training_data_format.goal_description, __o.training_data_format.goal_description)
        elif self.language == ProofAction.Language.LEAN:
            desc_cmp = DynamicLeanProofExecutor.goal_description_compare(self.training_data_format.goal_description, __o.training_data_format.goal_description)
        else:
            raise NotImplementedError(f"language {self.language} not supported")
        if desc_cmp == 0:
            return self.training_data_format >= __o.training_data_format
        else:
            return desc_cmp < 0

    def __le__(self, __o: object) -> bool:
        assert isinstance(__o, ProofState)
        assert self.language == __o.language, f"self.language: {self.language}, __o.language: {__o.language}"
        FailedProofState = FailedCoqProofState if self.language == ProofAction.Language.COQ else FailedLeanProofState
        if self == FailedProofState: # FailedProofState is the hardest state to reach
            return self.training_data_format == __o.training_data_format
        if __o == FailedProofState:
            return True
        assert isinstance(self.training_data_format, TrainingDataFormat)
        if self.language == ProofAction.Language.COQ:
            desc_cmp = DynamicCoqProofExecutor.goal_description_compare(self.training_data_format.goal_description, __o.training_data_format.goal_description)
        elif self.language == ProofAction.Language.LEAN:
            desc_cmp = DynamicLeanProofExecutor.goal_description_compare(self.training_data_format.goal_description, __o.training_data_format.goal_description)
        else:
            raise NotImplementedError(f"language {self.language} not supported")
        if desc_cmp == 0:
            return self.training_data_format <= __o.training_data_format
        else:
            return desc_cmp > 0
    
    def __lt__(self, __o: object) -> bool:
        assert isinstance(__o, ProofState)
        assert isinstance(self.training_data_format, TrainingDataFormat)
        assert self.language == __o.language, f"self.language: {self.language}, __o.language: {__o.language}"
        return self.training_data_format != __o.training_data_format and self.training_data_format <= __o.training_data_format
    
    def __gt__(self, __o: object) -> bool:
        assert isinstance(__o, ProofState)
        assert isinstance(self.training_data_format, TrainingDataFormat)
        assert self.language == __o.language, f"self.language: {self.language}, __o.language: {__o.language}"
        return self.training_data_format != __o.training_data_format and self.training_data_format >= __o.training_data_format


FailedCoqProofState = ProofState(training_data_format=None, language=ProofAction.Language.COQ)
FailedLeanProofState = ProofState(training_data_format=None, language=ProofAction.Language.LEAN)

if __name__ == "__main__":
    proof_state = ProofState(training_data_format=None)
    print(proof_state.serialize())
    pass