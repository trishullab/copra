#!/usr/bin/env python3

import sys
root_dir = f"{__file__.split('src')[0]}"
if root_dir not in sys.path:
    sys.path.append(root_dir)
import typing
import math
from collections import deque
from src.rl.abstraction import State, Action

class QInfo(object):
    def __init__(self, reward: float, done: bool, info: typing.Any, qval: float, global_info: typing.Any):
        self.reward = reward
        self.done = done
        self.info = info
        self.qval = qval
        self.global_info = global_info
    
class QTree(object):
    def __init__(self):
        self.root = None
        self.nodes : typing.Set[State] = dict()
        self.parents : typing.Dict[State, typing.Dict[Action, typing.Tuple[QInfo, State]]] = dict()
        self.edges : typing.Dict[State, typing.Dict[Action, typing.Tuple[QInfo, State]]] = dict()
    
    def state_in_tree(self, state: State) -> bool:
        return state in self.nodes

    def add(self, prev_state: State, action: Action, next_state: State, reward: float, done: bool, info: typing.Any):
        assert next_state is not None, f"next_state cannot be None"
        assert prev_state is not None, f"prev_state cannot be None"
        if len(self.nodes) == 0:
            assert len(self.parents) == 0 and len(self.edges) == 0, f"parents should be empty"
            # add root node
            self.root = prev_state
            self.nodes.add(prev_state)
            self.nodes.add(next_state)
            self.parents[next_state] = dict()
            self.edges[prev_state] = dict()
        else:
            assert self.root is not None, f"root cannot be None"
            assert prev_state in self.nodes, f"prev_state_node {prev_state} not in tree"
            if next_state not in self.nodes:
                self.nodes.add(next_state)
            if prev_state not in self.edges:
                self.edges[prev_state] = dict()
            if next_state not in self.parents:
                self.parents[next_state] = dict()
        qinfo = QInfo(reward, done, info, -math.inf, None)
        self.edges[prev_state][action] = (qinfo, next_state)
        self.parents[next_state][action] = (qinfo, prev_state)
    
    def get_all_ancestor_nodes(self, state: State) -> typing.List[typing.Tuple[int, QInfo, Action, State]]:
        assert state in self.nodes, f"node {node} not in tree"
        dq : typing.Deque[typing.Tuple[int, Action, QInfo, State]] = deque()
        dq.append((0, None, None, state))
        ancestors = []
        while len(dq) > 0:
            level, action, qinfo, node = dq.popleft()
            ancestors.append((level, action, qinfo, node))
            for action, (qinfo, parent) in self.parents[node].items():
                dq.append((level+1, action, qinfo, parent))
        return ancestors[1:]