#!/usr/bin/env python3

import sys

root_dir = f"{__file__.split('src')[0]}"
if root_dir not in sys.path:
    sys.path.append(root_dir)
import copy
import typing
from collections import deque
from src.rl.proof_tree import ProofTree
from src.rl.q_tree import QTreeStateInfo
from src.agent.gpt_guided_tree_search_policy import PromptSummary, ProofQTree, StateType, TreeSearchAction, TreeSearchActionType
from src.rl.simple_proof_env import ProgressState, ProofAction, ProofEnvInfo, ProofState
from src.agent.gpt_guided_tree_search_policy import ProofQInfo, ProofQTree
from src.rl.simple_proof_env import ProofAction, ProofEnvInfo, ProofState
from src.agent.gpt_guided_tree_search_policy import TreeSearchAlgorithm

class DFSTreeSearch(TreeSearchAlgorithm):
    def __init__(self):
        self._action_queue : deque = deque()
        pass

    def update_new_node(self, tree: ProofQTree, state: ProofState, action: ProofAction, next_state: ProofState, reward: float, done: bool, info: ProofEnvInfo):
        # The parent node had a similar next state and action pair.
        # Change all the nodes pointing to the parent as 'backtracked'
        should_add = True
        if tree.state_in_tree(state) and tree.state_in_tree(next_state) and action in tree.edges[state]:
            # It is possible that the next state is same but might be reachable from a different path.
            next_state_info = tree.edges[state][action]
            assert isinstance(next_state_info, QTreeStateInfo)
            assert next_state_info.state == next_state, f"next_state_info.state: {next_state_info.state}, next_state: {next_state} are not the same. even for the exact same action and state."
            should_add = False
            next_state_info.state_type = StateType.BACKTRACKED
            grandparent_node_infos = tree.parents[state].values()
            for grandparent_node_info in grandparent_node_infos:
                assert isinstance(grandparent_node_info, QTreeStateInfo)
                if grandparent_node_info.state != state:
                    # This is for grandparent nodes which are not the parent node. (self loop nodes)
                    for parent_node_action in tree.edges[grandparent_node_info.state]:
                        assert isinstance(parent_node_action, ProofAction)
                        parent_node_info = tree.edges[grandparent_node_info.state][parent_node_action]
                        assert isinstance(parent_node_info, QTreeStateInfo)
                        if parent_node_info.state == state:
                            # Mark all the edges from the grandparent node to the parent node as 'backtracked'
                            parent_proof_qinfo : ProofQInfo = parent_node_info.qinfo                    
                            parent_proof_qinfo.state_type = StateType.BACKTRACKED
            self._action_queue.append(TreeSearchAction(TreeSearchActionType.BACKTRACK, state, summary=None))
        if should_add:
            state_type = StateType.UNDISCOVERED
            if info.progress == ProgressState.FAILED:
                state_type = StateType.BACKTRACKED
            qinfo = ProofQInfo(reward, done, 0.0, proof_env_info=info, state_type=state_type)
            tree.add(state, action, next_state, qinfo)
            qinfo = copy.deepcopy(tree.edges[state][action].qinfo)
            qval = 1.0/qinfo.distance_from_root
            qinfo.qval = qval
            tree.update_qinfo(state, action, next_state, qinfo)
    
    def estimate_q_value(self, tree: ProofQTree, state: ProofState, action: ProofAction, next_state: ProofState, reward: float, done: bool, info: ProofEnvInfo) -> float:
        return super().estimate_q_value(tree, state, action, next_state, reward, done, info)
    
    def __call__(self, tree: ProofQTree, state: ProofState) -> TreeSearchAction:
        if len(self._action_queue) > 0:
            return self._action_queue.popleft()
        elif len(tree.nodes) == 0:
            qtree_state_info = QTreeStateInfo(state, 
                ProofQInfo(0.0, False, 0.0, has_loop=False, distance_from_root=0, proof_env_info=None, state_type=StateType.UNDISCOVERED))
            # There are no nodes in the tree, so we have to just give the summary from the proof state.
            return TreeSearchAction(TreeSearchActionType.NEXT_ACTION_SUMMARY_PROMPT, state, 
                summary=PromptSummary([], None, qtree_state_info))
        else:
            return self._dfs(tree, state)
    
    def _dfs(self, tree: ProofQTree, state: ProofState) -> TreeSearchAction:
        # 1. Get to the node which will act as the leaf node (has no cycle, has not failed, has no children, and is not visited)
        # (Step 1 ensures that we are doing a DFS search because we are going down the tree as far as possible)
        #   1.1 This means keep going down until you find a node which has no children, has not failed and has no cycle.
            #   1.1.1 Check if this node is 'done', if 'done' return 'Exit' action.
            #   1.1.2 If the node is NOT 'done', then generate the summary of the path taken to reach the node.
        # 2. If we are unable to find a node which has no children then try to find the first node that has failed and is not explored:
            #   1.2.1 Check if the node is 'Failed', if 'Failed' go to parent node and generate the summary of the path taken to reach the parent node. 
            #         Along with the failure summary.
            #   1.2.2 If the node is NOT 'Failed', then return then simply backtrack to the parent node.
        last_action : ProofAction = ProofAction(ProofAction.ActionType.NONE)
        old_actions : typing.List[ProofAction] = [ProofAction(ProofAction.ActionType.NONE)]
        stack = [(tree.root, old_actions)]
        found_leaf_node = False
        leaf_node = None
        found_cycle_node = False
        cycle_node_backtrack = None
        found_harder_node = False
        harder_node_backtrack = None
        backtracked_leaf_node = None
        found_backtracked_leaf_node = False
        incorrect_actions_from_node = []
        actions_till_now = []
        while len(stack) > 0 and not found_leaf_node and not found_backtracked_leaf_node and not found_cycle_node and not found_harder_node:
            state_info, old_actions = stack.pop()
            assert all([(old_action.action_type != ProofAction.ActionType.BACKTRACK and old_action.action_type != ProofAction.ActionType.EXIT) for old_action in old_actions])
            node : ProofState = state_info.state
            qinfo : ProofQInfo = state_info.qinfo
            if qinfo.state_type != StateType.BACKTRACKED:
                last_action = old_actions[-1]
                actions_till_now = old_actions[1:-1]
                # The condition above ensures that we do not visit any subtree coming 
                # from a node which has already been backtracked.
                if self._is_leaf_node(tree, node, qinfo, old_actions):
                    parent_state_info = None
                    if last_action.action_type != ProofAction.ActionType.NONE:
                        parent_state_info = tree.parents[node][last_action]
                    if parent_state_info is not None and \
                    parent_state_info.state <= node and \
                    last_action.action_type == ProofAction.ActionType.RUN_TACTIC: 
                        # This means that the new state is harder than the parent state and 
                        # hence we should not consider this state
                        found_harder_node = True
                        harder_node_backtrack = parent_state_info
                        qinfo.state_type = StateType.BACKTRACKED
                    else:
                        found_leaf_node = True
                        leaf_node = state_info
                elif self._has_all_backtracked_children(tree, node):
                    # This means that all the children of the node have been backtracked.
                    # We should backtrack to the parent node.
                    backtracked_leaf_node = state_info
                    found_backtracked_leaf_node = True
                    # Get all failed actions from this node
                    for action in tree.edges[node]:
                        incorrect_actions_from_node.append(action)
                # elif qinfo.has_self_loop and qinfo.proof_env_info.progress == ProgressState.FAILED:
                #     found_failed_node = True
                #     failed_node_backtrack = None
                #     if last_action.action_type != ProofAction.ActionType.NONE:
                #         failed_node_backtrack = tree.parents[node][last_action]
                #     qinfo.state_type = StateType.BACKTRACKED
                elif qinfo.has_loop and qinfo.proof_env_info.progress == ProgressState.RUNNING:
                    assert old_actions is not None and isinstance(old_actions, ProofAction)
                    assert last_action.action_type == ProofAction.ActionType.RUN_TACTIC, "Last action should be a tactic"
                    found_cycle_node = True
                    cycle_node_backtrack = state_info
                    looping_state : ProofState = qinfo.looping_state
                    assigned_qinfo = False
                    # Find the qinfo with the looping state
                    for action in tree.edges[node]:
                        child_state_info = tree.edges[node][action]
                        if child_state_info.state == looping_state:
                            assert isinstance(child_state_info.qinfo, ProofQInfo)
                            # This is the child state info where the lopping state is reached
                            child_state_info.qinfo.state_type = StateType.BACKTRACKED
                            assigned_qinfo = True
                            break
                    assert assigned_qinfo, "Could not find the child state info where the looping state is reached"
                else:
                    edges = tree.edges[node]
                    assert isinstance(state_info.qinfo, ProofQInfo)
                    # No need to filter nodes here because the condition for 'Backtracked' is already checked above
                    state_info_action_pairs = [(edges[action], old_actions + [action]) for action in edges]
                    # Sort the state info action pairs based on qval
                    state_info_action_pairs.sort(key=lambda x: x[0].qinfo.qval)
                    stack.extend(state_info_action_pairs)
                    qinfo.state_type = StateType.DISCOVERED
        if not found_leaf_node and not found_backtracked_leaf_node and not found_cycle_node and not found_harder_node:
            assert len(stack) == 0, "Stack should be empty"
            return TreeSearchAction(TreeSearchActionType.STOP, state, summary=None) # No need to check anymore coz we have exhausted our search
        else:
            # only one type of node can be found
            assert sum([found_leaf_node, found_backtracked_leaf_node, found_cycle_node, found_harder_node]) == 1, "Only one type of node can be found"
            assert last_action is not None, "Last action cannot be None"
            action_to_take : TreeSearchAction = None
            if found_leaf_node:
                action_to_take = TreeSearchAction(TreeSearchActionType.NEXT_ACTION_SUMMARY_PROMPT, state, summary=PromptSummary([], actions_till_now, last_action, leaf_node))
            elif found_backtracked_leaf_node:
                # No need to backtrack because we are already at the failed node, and we will automatically backtrack to the same state
                action_to_take = TreeSearchAction(TreeSearchActionType.FAILED_ACTION_SUMMARY_PROMPT, state, summary=PromptSummary(incorrect_actions_from_node, actions_till_now, last_action, backtracked_leaf_node))
            elif found_cycle_node:
                action_to_take = TreeSearchAction(TreeSearchActionType.BACKTRACK, state, summary=None)
                next_action = TreeSearchAction(TreeSearchActionType.CYCLIC_STATE_SUMMARY_PROMPT, state, summary=PromptSummary([last_action], actions_till_now, last_action, cycle_node_backtrack))
                self._action_queue.append(next_action)
            elif found_harder_node:
                action_to_take = TreeSearchAction(TreeSearchActionType.BACKTRACK, state, summary=None)
                next_action = TreeSearchAction(TreeSearchActionType.HARDER_STATE_SUMMARY_PROMPT, state, summary=PromptSummary([last_action], actions_till_now, last_action, harder_node_backtrack))
                self._action_queue.append(next_action)
            else:
                raise Exception("Should not reach here")
            return action_to_take
        
    def _is_leaf_node(self, tree: ProofQTree, state: ProofState, qinfo: ProofQInfo, last_action: ProofAction) -> bool:
        # A leaf node is a node which has no children or all its children are backtracked or it has a self loop and the action is get_dfns or get_thms
        return len(tree.edges[state]) == 0 or \
            (qinfo.has_self_loop and \
             qinfo.proof_env_info.progress == ProgressState.RUNNING and \
                (last_action.action_type == ProofAction.ActionType.GET_DFNS or \
                 last_action.action_type == ProofAction.ActionType.GET_THMS) )

    def _has_all_backtracked_children(self, tree: ProofQTree, state: ProofState) -> bool:
        return all([state_info.qinfo.state_type == StateType.BACKTRACKED for _, state_info in tree.edges[state].items()])