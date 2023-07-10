#!/usr/bin/env python3

import sys

root_dir = f"{__file__.split('src')[0]}"
if root_dir not in sys.path:
    sys.path.append(root_dir)
from src.rl.abstraction import Action
from enum import Enum


class ProofAction(Action):
    class Type(Enum):
        GET_THMS = 1
        GET_DFNS = 2
        RUN_TACTIC = 3
        BACKTRACK = 4

    def __init__(self, action_type: Type, **kwargs):
        assert isinstance(action_type, ProofAction.Type), f"action_type must be of type ProofAction.Type, not {type(action_type)}"
        self.action_type = action_type
        self.kwargs = kwargs
        if self.action_type == ProofAction.Type.RUN_TACTIC:
            assert "tactics" in self.kwargs, f"kwargs must contain a 'tactic' key for action_type {self.action_type}"
            assert isinstance(self.kwargs["tactics"], list), f"kwargs['tactics'] must be of type str, not {type(self.kwargs['tactics'])}"
            assert len(self.kwargs["tactics"]) > 0, f"kwargs['tactics'] must be a non-empty list"
            for tactic in self.kwargs["tactics"]:
                assert isinstance(tactic, str), f"kwargs['tactics'] must be of type str, not {type(tactic)}"
        else:
            assert len(self.kwargs) == 0, f"kwargs must be empty for action_type {self.action_type}"

    @property
    def name(self):
        return f"{self.action_type.name} | {self.kwargs}"

    def __call__(self):
        pass

    def serialize(self) -> str:
        return self.to_json()