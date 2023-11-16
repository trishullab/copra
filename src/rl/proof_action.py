#!/usr/bin/env python3

import sys

root_dir = f"{__file__.split('src')[0]}"
if root_dir not in sys.path:
    sys.path.append(root_dir)
import typing
from src.tools.coq_parse_utils import CoqLineByLineReader
from src.tools.lean_parse_utils import LeanLineByLineReader
from src.rl.abstraction import Action
from enum import Enum
from dataclasses import dataclass, field
from dataclasses_json import dataclass_json

@dataclass_json
@dataclass
class ProofAction(Action):
    class Language(Enum):
        COQ = 'COQ'
        LEAN = 'LEAN'

        def __str__(self):
            return self.name

    class ActionType(Enum):
        GET_DFNS_THMS = 'GET_DFNS_THMS'
        RUN_TACTIC = 'RUN_TACTIC'
        BACKTRACK = 'BACKTRACK'
        EXIT = 'EXIT'
        NONE = 'NONE'

        @staticmethod
        def get_order(action_type: 'ProofAction.ActionType'):
            if action_type == ProofAction.ActionType.EXIT:
                return 7
            elif action_type == ProofAction.ActionType.BACKTRACK:
                return 6
            elif action_type == ProofAction.ActionType.NONE:
                return 5
            if action_type == ProofAction.ActionType.RUN_TACTIC:
                return 4
            elif action_type == ProofAction.ActionType.GET_DFNS_THMS:
                return 3
            else:
                return 0

        def __str__(self):
            return self.name
        
        def __lt__(self, other):
            assert isinstance(other, ProofAction.ActionType), f"other must be of type ProofAction.ActionType, not {type(other)}"
            return ProofAction.ActionType.get_order(self) < ProofAction.ActionType.get_order(other)
        
        def __le__(self, other):
            assert isinstance(other, ProofAction.ActionType), f"other must be of type ProofAction.ActionType, not {type(other)}"
            return ProofAction.ActionType.get_order(self) <= ProofAction.ActionType.get_order(other)
        
        def __gt__(self, other):
            assert isinstance(other, ProofAction.ActionType), f"other must be of type ProofAction.ActionType, not {type(other)}"
            return ProofAction.ActionType.get_order(self) > ProofAction.ActionType.get_order(other)
        
        def __ge__(self, other):
            assert isinstance(other, ProofAction.ActionType), f"other must be of type ProofAction.ActionType, not {type(other)}"
            return ProofAction.ActionType.get_order(self) >= ProofAction.ActionType.get_order(other)

    action_type: ActionType
    language: Language
    kwargs: typing.Optional[dict] = field(default_factory=dict)
    def __init__(self, action_type: ActionType, language: Language, **kwargs):
        assert isinstance(action_type, ProofAction.ActionType), f"action_type must be of type ProofAction.Type, not {type(action_type)}"
        self.action_type = action_type
        self.language = language
        if kwargs is not None and isinstance(kwargs, dict) and len(kwargs) == 1 and "kwargs" in kwargs:
            kwargs = kwargs["kwargs"] # TODO: this is a hack to get around the fact that dataclasses_json doesn't support having parameterized fields in the constructor
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
            if self.language == ProofAction.Language.COQ:
                all_tactics = '\n'.join(self.kwargs['tactics'])
                reader = CoqLineByLineReader(file_content=all_tactics)
                tactics = list(reader.instruction_step_generator())
                self.kwargs['tactics'] = tactics
            elif self.language == ProofAction.Language.LEAN:
                all_tactics = '\n'.join(self.kwargs['tactics'])
                # reader = LeanLineByLineReader(file_content=all_tactics)
                tactics = [all_tactics]
                self.kwargs['tactics'] = tactics
        self.original_message : typing.Any = None
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

    def __ge__(self, other):
        if not isinstance(other, ProofAction):
            return False
        return self.action_type >= other.action_type

    def __gt__(self, other):
        if not isinstance(other, ProofAction):
            return False
        return self.action_type > other.action_type
    
    def __le__(self, other):
        if not isinstance(other, ProofAction):
            return False
        return self.action_type <= other.action_type
    
    def __lt__(self, other):
        if not isinstance(other, ProofAction):
            return False
        return self.action_type < other.action_type

    def __call__(self):
        pass

    def serialize(self) -> str:
        return self.to_json()

if __name__ == "__main__":
    action = ProofAction(action_type=ProofAction.ActionType.RUN_TACTIC, tactics=["intros.", "reflexivity."])
    print(action.serialize())
    action1 = ProofAction.schema().loads(action.serialize())
    assert action == action1