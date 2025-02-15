#!/usr/bin/env python3

import sys
root_dir = f"{__file__.split('src')[0]}"
if root_dir not in sys.path:
    sys.path.append(root_dir)
import typing
import math
from src.rl.abstraction import QFunction, State, Action
from src.rl.proof_env import ProofEnvInfo
from src.rl.q_tree import QGraph

class ProofQFunction(QFunction):
    def __init__(self):
        super().__init__()
        self.q_tree = QGraph()

    def __call__(self, state: State, action: Action) -> typing.Tuple[float, ProofEnvInfo]:
        if self.q_tree.state_in_tree(state):
            qinfo, _ = self.q_tree.edges[state][action]
            return qinfo.qval, qinfo.global_info
        return -math.inf, None

    def update(self, state: State, action: Action, next_state: State, reward: float, done: bool, info: ProofEnvInfo):
        raise NotImplementedError