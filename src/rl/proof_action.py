#!/usr/bin/env python3

import sys

root_dir = f"{__file__.split('src')[0]}"
if root_dir not in sys.path:
    sys.path.append(root_dir)
from src.tools.coq_parse_utils import CoqLineByLineReader
from src.rl.abstraction import Action
from enum import Enum


class ProofAction(Action):
    class ActionType(Enum):
        GET_THMS = 1
        GET_DFNS = 2
        RUN_TACTIC = 3
        BACKTRACK = 4
        EXIT = 5

    def __init__(self, action_type: ActionType, **kwargs):
        assert isinstance(action_type, ProofAction.ActionType), f"action_type must be of type ProofAction.Type, not {type(action_type)}"
        self.action_type = action_type
        self.kwargs = kwargs
        if self.action_type == ProofAction.ActionType.RUN_TACTIC:
            assert "tactics" in self.kwargs, f"kwargs must contain a 'tactic' key for action_type {self.action_type}"
            assert isinstance(self.kwargs["tactics"], list), f"kwargs['tactics'] must be of type str, not {type(self.kwargs['tactics'])}"
            assert len(self.kwargs["tactics"]) > 0, f"kwargs['tactics'] must be a non-empty list"
            for tactic in self.kwargs["tactics"]:
                assert isinstance(tactic, str), f"kwargs['tactics'] must be of type str, not {type(tactic)}"
        else:
            assert len(self.kwargs) == 0, f"kwargs must be empty for action_type {self.action_type}"
        self._post_init()
    
    def _post_init(self):
        type = self.action_type
        if type == ProofAction.ActionType.RUN_TACTIC:
            all_tactics = '\n'.join(self.kwargs['tactics'])
            reader = CoqLineByLineReader(file_content=all_tactics)
            tactics = list(reader.instruction_step_generator())
            self.kwargs['tactics'] = tactics
        pass

    @property
    def name(self):
        return f"{self.action_type.name} | {self.kwargs}"
    
    def __eq__(self, other):
        if not isinstance(other, ProofAction):
            return False
        if self.action_type != other.action_type:
            return False
        if self.action_type == ProofAction.ActionType.RUN_TACTIC:
            tactics = self.kwargs['tactics']
            other_tactics = other.kwargs['tactics']
            if len(tactics) != len(other_tactics):
                return False
            for i in range(len(tactics)):
                if tactics[i] != other_tactics[i]:
                    return False
        return True
    
    def __hash__(self):
        if self.action_type != ProofAction.ActionType.RUN_TACTIC:
            return hash(self.action_type)
        else:
            tactics = self.kwargs['tactics']
            return hash((self.action_type, tuple(tactics)))

    def __call__(self):
        pass

    def serialize(self) -> str:
        return f"""
{{
    'action_type': '{self.action_type.name}',
    'kwargs': {self.kwargs}
}}
"""