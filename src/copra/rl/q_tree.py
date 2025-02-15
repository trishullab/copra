#!/usr/bin/env python3

import typing
import copy
from collections import OrderedDict, deque
from itp_interface.rl.abstraction import State, Action
from dataclasses import dataclass
from dataclasses_json import dataclass_json

@dataclass_json
@dataclass
class QInfo(object):
    reward: float
    done: bool
    qval: float
    has_loop: bool = False
    has_self_loop: bool = False
    distance_from_root: int = -1

    def _post_init_(self):
        self.looping_state : State = None

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

class QTreeStateInfo(object):
    def __init__(self, state: State, qinfo: QInfo):
        self.state = state
        self.qinfo = qinfo

class QGraph(object):
    """
    This is a directed graph and not a tree. However, this graph has a root and all nodes are reachable from the root.
    """
    def __init__(self):
        self.root: typing.Optional[QTreeStateInfo] = None
        self.nodes : typing.Set[State] = set()
        self.parents : typing.OrderedDict[State, typing.Dict[Action, QTreeStateInfo]] = OrderedDict()
        self.edges : typing.OrderedDict[State, typing.Dict[Action, QTreeStateInfo]] = OrderedDict()
    
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
            root_qinfo = copy.deepcopy(qinfo)
            root_qinfo.distance_from_root = 0
            self.root = QTreeStateInfo(state=prev_state_copy, qinfo=root_qinfo)
            self.nodes.add(prev_state_copy)
            self.nodes.add(next_state_copy)
            qinfo_copy.distance_from_root = 1
            self.parents[next_state_copy] = dict()
            self.parents[prev_state_copy] = dict() # root has no parent
            self.edges[prev_state_copy] = dict()
            self.edges[next_state_copy] = dict() # no edges going out of next state
        else:
            assert self.root is not None, f"root cannot be None"
            assert prev_state in self.nodes, f"prev_state_node {prev_state} not in tree"
            assert prev_state in self.edges, f"prev_state_node {prev_state} not in tree edges"
            assert prev_state in self.parents, f"prev_state_node {prev_state} not in tree parents"
            if next_state not in self.nodes:
                self.nodes.add(next_state_copy)
            if next_state not in self.edges:
                self.edges[next_state_copy] = dict()
            if next_state not in self.parents:
                self.parents[next_state_copy] = dict()
            prev_dist = [state_info.qinfo.distance_from_root for state_info in self.parents[prev_state_copy].values()]
            if len(prev_dist) > 0:
                parent_distance_from_root = min(prev_dist)
            else:
                parent_distance_from_root = 0
            qinfo_copy.distance_from_root = parent_distance_from_root + 1

        self.edges[prev_state_copy][action_copy] = QTreeStateInfo(next_state_copy, qinfo_copy)
        self.parents[next_state_copy][action_copy] = QTreeStateInfo(prev_state_copy, qinfo_copy)
        qinfo_copy.has_self_loop = self._has_self_loop(next_state_copy)
        if not qinfo_copy.has_self_loop:
            qinfo_copy.has_loop, qinfo_copy.looping_state = self._has_any_loop(next_state_copy)
        else:
            qinfo_copy.has_loop = True
            qinfo_copy.looping_state = next_state_copy
   
    def is_leaf(self, state: State) -> bool:
        assert state in self.nodes, "state not in the tree"
        return len(self.edges[state]) == 0 # There should be no outgoing edge
    
    def _has_self_loop(self, state: State) -> bool:
        assert state in self.nodes, "state not in the tree"
        parents = self.parents[state]
        for _, state_info in parents.items():
            parent = state_info.state
            if parent == state:
                return True
        return False
    
    def _has_any_loop(self, state: State) -> bool:
        assert state in self.nodes, "state not in the tree"
        visited = set()
        dq = deque()
        dq.append(state)
        trajectory : typing.List[State] = []
        has_loop = False
        while len(dq) > 0 and not has_loop:
            node = dq.popleft()
            trajectory.append(node)
            if node in visited:
                has_loop = True
            else:
                visited.add(node)
                for _, state_info in self.parents[node].items():
                    parent = state_info.state
                    dq.append(parent)
        looping_state = trajectory[-1] if has_loop else None
        return has_loop, looping_state

    def update_qinfo(self, prev_state: State, action: Action, next_state: State, new_qinfo: QInfo):
        assert prev_state in self.nodes, f"prev_state_node {prev_state} not in tree"
        assert next_state in self.nodes, f"next_state_node {next_state} not in tree"
        assert prev_state in self.edges, f"prev_state_node {prev_state} not in tree"
        assert next_state in self.parents, f"next_state_node {next_state} not in tree"
        actual_next_state_info = self.edges[prev_state][action]
        actual_prev_state_info = self.parents[next_state][action]
        assert actual_next_state_info.state == next_state, f"next_state {next_state} not in tree"
        assert actual_prev_state_info.state == prev_state, f"prev_state {prev_state} not in tree"
        qinfo_copy = copy.deepcopy(new_qinfo)
        self.edges[actual_prev_state_info.state][action] = QTreeStateInfo(actual_next_state_info.state, qinfo_copy)
        self.parents[actual_next_state_info.state][action] = QTreeStateInfo(actual_prev_state_info.state, qinfo_copy)
    
    def get_all_ancestor_nodes(self, state: State) -> typing.List[typing.Tuple[int, QInfo, Action, State]]:
        assert state in self.nodes, f"node {node} not in tree"
        dq : typing.Deque[typing.Tuple[int, Action, QInfo, State]] = deque()
        dq.append((0, None, None, state))
        ancestors = []
        while len(dq) > 0:
            level, action, qinfo, node = dq.popleft()
            ancestors.append((level, action, qinfo, node))
            for action, state_info in self.parents[node].items():
                qinfo = state_info.qinfo
                parent = state_info.state
                dq.append((level+1, action, qinfo, parent))
        return ancestors[1:]
    
    def serialize(self) -> str:
        # Conver to QTreeNodes
        qtree_nodes = []
        for prev_state, edges in self.edges.items():
            actions = []
            next_states = []
            qinfos = []
            for action, state_info in edges.items():
                qinfo = state_info.qinfo
                next_state = state_info.state
                actions.append(action)
                next_states.append(next_state)
                qinfos.append(qinfo)
            qtree_nodes.append(QTreeNode(prev_state, actions, next_states, qinfos))
        return QTreeNode.schema().dumps(qtree_nodes, many=True)
    
    @staticmethod
    def deserialize(data: str) -> 'QGraph':
        qtree_nodes : typing.List[QTreeNode] = QTreeNode.schema().loads(data, many=True)
        qtree = QGraph()
        for qtree_node in qtree_nodes:
            for action, next_state, qinfo in zip(qtree_node.actions, qtree_node.next_state, qtree_node.qinfo):
                qtree.add(qtree_node.prev_state, action, next_state, qinfo)
        return qtree