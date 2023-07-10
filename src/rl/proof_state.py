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
    def __init__(self, training_data_format: TrainingDataFormat):
        assert isinstance(training_data_format, TrainingDataFormat), f"training_data_format must be of type TrainingDataFormat, not {type(training_data_format)}"
        self.training_data_format = training_data_format

    @property
    def name(self):
        return f"{self.goal.name} | {self.proof.name}"

    def __eq__(self, other):
        return self.name == other.name

    def __hash__(self):
        return hash(self.name)

    def serialize(self) -> str:
        return self.to_json()