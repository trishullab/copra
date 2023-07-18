#!/usr/bin/env python3

import sys

root_dir = f"{__file__.split('src')[0]}"
if root_dir not in sys.path:
    sys.path.append(root_dir)
import typing
from abc import ABC, abstractmethod

from src.prompt_generator.coq_gpt_grammar import CoqGPTRequestGrammar, CoqGPTResponseGrammar

class Prompter(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def generate_prompt(self, state: State, action: Action) -> str:
        pass

    @abstractmethod
    def parse_response(self, response: str) -> typing.Tuple[State, float, bool, dict]:
        pass