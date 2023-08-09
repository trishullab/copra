#!/usr/bin/env python3

import sys
root_dir = f"{__file__.split('src')[0]}"
if root_dir not in sys.path:
    sys.path.append(root_dir)
import typing
import copy
from collections import deque
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
        if tree.state_in_tree(state) and tree.state_in_tree(next_state):
            next_state_info = tree.edges[state][action]
            assert isinstance(next_state_info, ProofQInfo)
            if next_state_info.state == next_state:
                should_add = False
                next_state_info.state_type = StateType.BACKTRACKED
                grandparent_node_infos = tree.parents[state].values()
                for grandparent_node_info in grandparent_node_infos:
                    assert isinstance(grandparent_node_info, ProofQInfo)
                    proof_qinfo : ProofQInfo = grandparent_node_info.qinfo
                    proof_qinfo.state_type = StateType.BACKTRACKED
        if should_add:
            qinfo = ProofQInfo(reward, done, 0.0, proof_env_info=info, state_type=StateType.UNDISCOVERED)
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
            # There are no nodes in the tree, so we have to just give the summary from the proof state.
            return TreeSearchAction(TreeSearchActionType.NEXT_ACTION_SUMMARY_PROMPT, summary=PromptSummary([], state))
        else:
            return self._dfs(tree)
    
    def _dfs(self, tree: ProofQTree):
        # 1. Get to the node which will act as the leaf node (has no cycle, has not failed, has no children, and is not visited)
        # (Step 1 ensures that we are doing a DFS search because we are going down the tree as far as possible)
        #   1.1 This means keep going down until you find a node which has no children, has not failed and has no cycle.
            #   1.1.1 Check if this node is 'done', if 'done' return 'Exit' action.
            #   1.1.2 If the node is NOT 'done', then generate the summary of the path taken to reach the node.
        # 2. If we are unable to find a node which has no children then try to find the first node that has failed and is not explored:
            #   1.2.1 Check if the node is 'Failed', if 'Failed' go to parent node and generate the summary of the path taken to reach the parent node. 
            #         Along with the failure summary.
            #   1.2.2 If the node is NOT 'Failed', then return then simply backtrack to the parent node.
        last_action : typing.Optional[ProofAction] = None
        stack = [(tree.root, None)]
        found_leaf_node = False
        leaf_node = None
        found_failed_node = False
        failed_node_backtrack = None
        found_cycle_node = False
        cycle_node_backtrack = None
        found_harder_node = False
        harder_node_backtrack = None
        while len(stack) > 0 and not found_leaf_node and not found_failed_node and not found_cycle_node and not found_harder_node:
            state_info, old_action = stack.pop()
            node : ProofState = state_info.state
            qinfo : ProofQInfo = state_info.qinfo
            if qinfo.state_type != StateType.BACKTRACKED:
                if tree.is_leaf(node):
                    last_action = old_action
                    parent = tree.parents[node][last_action]
                    if parent <= node: # This means that the new state is harder than the parent state and hence we should not consider this state
                        found_harder_node = True
                        harder_node_backtrack = parent
                        qinfo.state_type = StateType.BACKTRACKED
                    else:
                        found_leaf_node = True
                        leaf_node = node
                elif qinfo.has_self_loop and qinfo.proof_env_info.progress == ProgressState.FAILED:
                    found_failed_node = True
                    last_action = old_action
                    failed_node_backtrack = tree.parents[node][last_action]
                    qinfo.state_type = StateType.BACKTRACKED
                elif qinfo.has_loop and qinfo.proof_env_info.progress == ProgressState.RUNNING:
                    found_cycle_node = True
                    last_action = old_action
                    cycle_node_backtrack = tree.parents[node][last_action]
                    qinfo.state_type = StateType.BACKTRACKED
                else:
                    edges = tree.edges[node]
                    state_info_action_pairs = [(edges[action], action) for action in edges if state_info.qinfo.state_type == StateType.UNDISCOVERED]
                    # Sort the state info action pairs based on qval
                    state_info_action_pairs.sort(key=lambda x: x[0].qinfo.qval)
                    stack.extend(state_info_action_pairs)
                    qinfo.state_type = StateType.DISCOVERED
        if not found_leaf_node and not found_failed_node and not found_cycle_node and not found_harder_node:
            assert len(stack) == 0, "Stack should be empty"
            return TreeSearchAction(TreeSearchActionType.STOP, summary=None) # No need to check anymore coz we have exhausted our search
        else:
            # only one type of node can be found
            assert sum([found_leaf_node, found_failed_node, found_cycle_node, found_harder_node]) == 1, "Only one type of node can be found"
            assert last_action is not None, "Last action cannot be None"
            action_to_take : TreeSearchAction = None
            if found_leaf_node:
                action_to_take = TreeSearchAction(TreeSearchActionType.NEXT_ACTION_SUMMARY_PROMPT, summary=PromptSummary([], leaf_node))
            elif found_failed_node:
                action_to_take = TreeSearchAction(TreeSearchActionType.BACKTRACK, summary=None)
                next_action = TreeSearchAction(TreeSearchActionType.FAILED_ACTION_SUMMARY_PROMPT, summary=PromptSummary([last_action], failed_node_backtrack))
                self._action_queue.append(next_action)
            elif found_cycle_node:
                action_to_take = TreeSearchAction(TreeSearchActionType.BACKTRACK, summary=None)
                next_action = TreeSearchAction(TreeSearchActionType.CYCLIC_STATE_SUMMARY_PROMPT, summary=PromptSummary([last_action], cycle_node_backtrack))
                self._action_queue.append(next_action)
            elif found_harder_node:
                action_to_take = TreeSearchAction(TreeSearchActionType.BACKTRACK, summary=None)
                next_action = TreeSearchAction(TreeSearchActionType.HARDER_STATE_SUMMARY_PROMPT, summary=PromptSummary([last_action], harder_node_backtrack))
                self._action_queue.append(next_action)
            else:
                raise Exception("Should not reach here")
            return action_to_take