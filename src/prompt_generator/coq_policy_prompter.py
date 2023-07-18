#!/usr/bin/env python3

import sys
root_dir = f"{__file__.split('src')[0]}"
if root_dir not in sys.path:
    sys.path.append(root_dir)

import typing
from src.rl.abstraction import Action, Env
from src.prompt_generator.prompter import PolicyPrompter

class CoqPolicyPrompter(PolicyPrompter):
    def __init__(self):
        pass

    def generate_prompt(self, env: Env) -> str:
        pass

    def parse_response(self, response: str) -> typing.Tuple[Action, float]:
        pass