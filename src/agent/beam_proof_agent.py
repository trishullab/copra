#!/usr/bin/env python3

import sys
from prompt_generator.prompter import PolicyPrompter
from rl.abstraction import Env
root_dir = f"{__file__.split('src')[0]}"
if root_dir not in sys.path:
    sys.path.append(root_dir)

from src.rl.abstraction import Env
from src.prompt_generator.prompter import PolicyPrompter
from src.agent.proof_agent import ProofAgent

class BeamProofAgent(ProofAgent):
    def __init__(self, beam_size: int):
        self.beam_size = beam_size
    
    def prove(self, env: Env, prompter: PolicyPrompter, max_prompts: int = 100) -> bool:
        # Implement beam search here
        pass