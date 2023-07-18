#!/usr/bin/env python3

import sys
root_dir = f"{__file__.split('src')[0]}"
if root_dir not in sys.path:
    sys.path.append(root_dir)

from abc import ABC, abstractmethod
from src.rl.abstraction import Env
from src.prompt_generator.prompter import PolicyPrompter


class ProofAgent(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def prove(self, env: Env, prompter: PolicyPrompter, max_prompts: int = 100) -> bool:
        pass