#!/usr/bin/env python3

import sys

root_dir = f"{__file__.split('src')[0]}"
if root_dir not in sys.path:
    sys.path.append(root_dir)
import typing
from src.rl.abstraction import State, Action, Env


class ProofEnv(Env):
    def __init__(self):
        pass

    def reset(self):
        pass

    def step(self, action: Action) -> typing.Tuple[State, float, bool, dict]:
        pass