#!/usr/bin/env python3

import sys
root_dir = f"{__file__.split('src')[0]}"
if root_dir not in sys.path:
    sys.path.append(root_dir)
import typing
import copy
from collections import deque
from src.rl.abstraction import State, Action
from dataclasses import dataclass, field
from dataclasses_json import dataclass_json

@dataclass_json
@dataclass
class QInfo(object):
    reward: float
    done: bool
    qval: float

    def serialize(self) -> str:
        return self.to_json()
    
    @staticmethod
    def deserialize(data: str) -> 'QInfo':
        return QInfo.schema().loads(data)

@dataclass_json
@dataclass
class QTreeNode(object):
    prev_state: State
    actions: typing.List[Action]
    next_state: typing.List[State]
    qinfo: typing.List[QInfo]

    def serialize(self) -> str:
        return self.to_json()
    
    @staticmethod
    def deserialize(data: str) -> 'QTreeNode':
        return QTreeNode.schema().loads(data)
    
class QTree(object):
    def __init__(self):
        self.root: typing.Optional[QInfo] = None
        self.nodes : typing.Set[State] = dict()
        self.parents : typing.Dict[State, typing.Dict[Action, typing.Tuple[QInfo, State]]] = dict()
        self.edges : typing.Dict[State, typing.Dict[Action, typing.Tuple[QInfo, State]]] = dict()
    
    def state_in_tree(self, state: State) -> bool:
        return state in self.nodes

    def add(self, prev_state: State, action: Action, next_state: State, qinfo: QInfo):
        assert next_state is not None, f"next_state cannot be None"
        assert prev_state is not None, f"prev_state cannot be None"
        qinfo_copy = copy.deepcopy(qinfo)
        next_state_copy = copy.deepcopy(next_state)
        prev_state_copy = copy.deepcopy(prev_state)
        action_copy = copy.deepcopy(action)
        if len(self.nodes) == 0:
            assert len(self.parents) == 0 and len(self.edges) == 0, f"parents should be empty"
            # add root node
            self.root = prev_state_copy
            self.nodes.add(prev_state_copy)
            self.nodes.add(next_state_copy)
            self.parents[next_state_copy] = dict()
            self.edges[prev_state_copy] = dict()
        else:
            assert self.root is not None, f"root cannot be None"
            assert prev_state in self.nodes, f"prev_state_node {prev_state} not in tree"
            if next_state not in self.nodes:
                self.nodes.add(next_state_copy)
            if prev_state not in self.edges:
                self.edges[prev_state_copy] = dict()
            if next_state not in self.parents:
                self.parents[next_state_copy] = dict()

        self.edges[prev_state_copy][action_copy] = (qinfo_copy, next_state_copy)
        self.parents[next_state_copy][action_copy] = (qinfo_copy, prev_state_copy)
    
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
    
    def serialize(self) -> str:
        # Conver to QTreeNodes
        qtree_nodes = []
        for prev_state, edges in self.edges.items():
            actions = []
            next_states = []
            qinfos = []
            for action, (qinfo, next_state) in edges.items():
                actions.append(action)
                next_states.append(next_state)
                qinfos.append(qinfo)
            qtree_nodes.append(QTreeNode(prev_state, actions, next_states, qinfos))
        return QTreeNode.schema().dumps(qtree_nodes, many=True)
    
    @staticmethod
    def deserialize(data: str) -> 'QTree':
        qtree_nodes : typing.List[QTree] = QTreeNode.schema().loads(data, many=True)
        qtree = QTree()
        for qtree_node in qtree_nodes:
            for action, next_state, qinfo in zip(qtree_node.actions, qtree_node.next_state, qtree_node.qinfo):
                qtree.add(qtree_node.prev_state, action, next_state, qinfo)
        return qtree