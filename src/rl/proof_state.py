#!/usr/bin/env python3

import sys

root_dir = f"{__file__.split('src')[0]}"
if root_dir not in sys.path:
    sys.path.append(root_dir)
from src.rl.abstraction import State
from src.tools.training_data_format import TrainingDataFormat
from dataclasses_json import dataclass_json
from dataclasses import dataclass

@dataclass_json
@dataclass
class ProofState(State):
    training_data_format: TrainingDataFormat

    @property
    def name(self):
        return f"{self.training_data_format.start_goals}"[:25]

    def __eq__(self, other):
        if not isinstance(other, ProofState):
            return False
        return self.training_data_format == other.training_data_format

    def __hash__(self):
        return hash(self.training_data_format)

    def serialize(self) -> str:
        serialized_json = self.to_json()
        return serialized_json
    
    def __ge__(self, __o: object) -> bool:
        assert isinstance(__o, ProofState)
        self.training_data_format >= __o.training_data_format

    def __le__(self, __o: object) -> bool:
        assert isinstance(__o, ProofState)
        self.training_data_format <= __o.training_data_format
    
    def __lt__(self, __o: object) -> bool:
        assert isinstance(__o, ProofState)
        return self.training_data_format != __o.training_data_format and self.training_data_format <= __o.training_data_format
    
    def __gt__(self, __o: object) -> bool:
        assert isinstance(__o, ProofState)
        return self.training_data_format != __o.training_data_format and self.training_data_format >= __o.training_data_format