#!/usr/bin/env python3

import sys

root_dir = f"{__file__.split('src')[0]}"
if root_dir not in sys.path:
    sys.path.append(root_dir)
import copy
import typing
from collections import deque
from src.rl.q_tree import QTreeStateInfo
from src.agent.gpt_guided_tree_search_policy import FailureReason, PromptSummary, ProofQTree, StateType, TreeSearchAction, TreeSearchActionType
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
            assert isinstance(next_state_info.qinfo, ProofQInfo)
            should_add = False
            assert next_state_info.qinfo.proof_env_info.progress != ProgressState.RUNNING, "The next state should not be running as DFS allows only one path to run"
            next_state_info.qinfo.state_type = StateType.BACKTRACKED
            grandparent_node_infos = tree.parents[state].values()
            found_parent_node = False
            for grandparent_node_info in grandparent_node_infos:
                assert isinstance(grandparent_node_info, QTreeStateInfo)
                # if grandparent_node_info.state != state:
                    # This is for grandparent nodes which are not the parent node. (self loop nodes)
                for parent_node_action in tree.edges[grandparent_node_info.state]:
                    assert isinstance(parent_node_action, ProofAction)
                    parent_node_info = tree.edges[grandparent_node_info.state][parent_node_action]
                    assert isinstance(parent_node_info, QTreeStateInfo)
                    if parent_node_info.state == state:
                        # Mark all the edges from the grandparent node to the parent node as 'backtracked'
                        parent_proof_qinfo : ProofQInfo = parent_node_info.qinfo                    
                        parent_proof_qinfo.state_type = StateType.BACKTRACKED
                        if parent_proof_qinfo.proof_env_info.progress == ProgressState.RUNNING:
                            parent_proof_qinfo.proof_env_info.progress = ProgressState.FAILED
                            parent_proof_qinfo.proof_env_info.error_message = "This tactic fails because it leads to proof-state which eventually fails."
                            parent_proof_qinfo.failure_reason = FailureReason.SUBSEQUENT_STATE_FAILED
                            found_parent_node = True
            if found_parent_node:
                self._action_queue.append(TreeSearchAction(TreeSearchActionType.BACKTRACK, grandparent_node_info.state, summary=None))
        if should_add:
            state_type = StateType.UNDISCOVERED
            if info.progress == ProgressState.FAILED:
                state_type = StateType.BACKTRACKED
            qinfo = ProofQInfo(reward, done, 0.0, proof_env_info=info, state_type=state_type)
            tree.add(state, action, next_state, qinfo)
            qinfo : ProofQInfo = copy.deepcopy(tree.edges[state][action].qinfo)
            # Check if this node has a loop
            if qinfo.proof_env_info.progress == ProgressState.RUNNING:
                parent_node_info = tree.parents[next_state][action]
                if qinfo.has_loop:
                    qinfo.state_type = StateType.BACKTRACKED
                    self._action_queue.append(TreeSearchAction(TreeSearchActionType.BACKTRACK, state, summary=None))
                    # Update the qval of the parent node
                    qinfo.qval = -0.5
                    qinfo.proof_env_info.progress = ProgressState.FAILED
                    qinfo.proof_env_info.error_message = "This tactic fails becuase it does NOT simplify the goal, and takes us to a goal which we have already seen."
                    qinfo.failure_reason = FailureReason.CYCLIC_STATE
                elif parent_node_info.state <= next_state:
                    qinfo.state_type = StateType.BACKTRACKED
                    self._action_queue.append(TreeSearchAction(TreeSearchActionType.BACKTRACK, state, summary=None))
                    # Update the qval of the parent node
                    qinfo.qval = -0.5
                    qinfo.proof_env_info.progress = ProgressState.FAILED
                    qinfo.proof_env_info.error_message = "This tactic fails because it does NOT simplify the goal, and takes us to a goal which is harder (or as hard) as the current goal."
                    qinfo.failure_reason = FailureReason.HARDER_STATE
                else:
                    qval = 1.0/qinfo.distance_from_root
                    qinfo.qval = qval
                    qinfo.failure_reason = FailureReason.NONE
            else:
                qinfo.qval = -0.5
                qinfo.failure_reason = FailureReason.COMPILE_FAILED
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
                summary=PromptSummary([], [], None, qtree_state_info))
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
        found_backtracked_leaf_node = False
        incorrect_actions_from_node = []
        actions_till_now = []
        action_to_take : TreeSearchAction = None
        while len(stack) > 0 and not found_leaf_node and not found_backtracked_leaf_node:
            state_info, old_actions = stack.pop()
            assert all([(old_action.action_type != ProofAction.ActionType.BACKTRACK and old_action.action_type != ProofAction.ActionType.EXIT) for old_action in old_actions])
            node : ProofState = state_info.state
            qinfo : ProofQInfo = state_info.qinfo
            if qinfo.state_type != StateType.BACKTRACKED:
                last_action = old_actions[-1]
                actions_till_now = old_actions[1:-1]
                # The condition above ensures that we do not visit any subtree coming 
                # from a node which has already been backtracked.
                if self._is_leaf_node(tree, node, qinfo, last_action):
                    found_leaf_node = True
                    leaf_node = state_info
                    action_to_take = TreeSearchAction(TreeSearchActionType.NEXT_ACTION_SUMMARY_PROMPT, node, summary=PromptSummary([], actions_till_now, last_action, leaf_node))
                elif self._has_all_backtracked_children(tree, node):
                    # This means that all the children of the node have been backtracked.
                    # We should backtrack to the parent node.
                    found_backtracked_leaf_node = True
                    last_backtracked_action = next(reversed(tree.edges[node]))
                    incorrect_actions_from_node = list(iter(tree.edges[node]))[:-1]
                    qinfo = tree.edges[node][last_backtracked_action].qinfo
                    last_node_info = tree.edges[node][last_backtracked_action]
                    if qinfo.failure_reason == FailureReason.SUBSEQUENT_STATE_FAILED:
                        action_to_take = TreeSearchAction(TreeSearchActionType.FAILED_ACTION_SUMMARY_PROMPT, node, 
                            summary=PromptSummary(
                            incorrect_actions_from_node, 
                            actions_till_now + [last_action], 
                            last_backtracked_action, 
                            last_node_info))
                    elif qinfo.failure_reason == FailureReason.CYCLIC_STATE:
                        action_to_take = TreeSearchAction(TreeSearchActionType.CYCLIC_STATE_SUMMARY_PROMPT, node,
                            summary=PromptSummary(
                            incorrect_actions_from_node,
                            actions_till_now + [last_action],
                            last_backtracked_action,
                            last_node_info))
                    elif qinfo.failure_reason == FailureReason.HARDER_STATE:
                        action_to_take = TreeSearchAction(TreeSearchActionType.HARDER_STATE_SUMMARY_PROMPT, node,
                            summary=PromptSummary(
                            incorrect_actions_from_node,
                            actions_till_now + [last_action],
                            last_backtracked_action,
                            last_node_info))
                    elif qinfo.failure_reason == FailureReason.COMPILE_FAILED:
                        action_to_take = TreeSearchAction(TreeSearchActionType.FAILED_ACTION_SUMMARY_PROMPT, node,
                            summary=PromptSummary(
                            incorrect_actions_from_node,
                            actions_till_now + [last_action],
                            last_backtracked_action,
                            last_node_info))
                    else:
                        raise ValueError(f"Unknown failure reason: {qinfo.failure_reason}")
                else:
                    edges = tree.edges[node]
                    assert isinstance(state_info.qinfo, ProofQInfo)
                    # No need to filter nodes here because the condition for 'Backtracked' is already checked above
                    state_info_action_pairs = [(edges[action], old_actions + [action]) for action in edges]
                    # Sort the state info action pairs based on qval
                    state_info_action_pairs.sort(key=lambda x: x[0].qinfo.qval)
                    stack.extend(state_info_action_pairs)
                    qinfo.state_type = StateType.DISCOVERED
        if not found_leaf_node and not found_backtracked_leaf_node:
            assert len(stack) == 0, "Stack should be empty"
            return TreeSearchAction(TreeSearchActionType.STOP, state, summary=None) # No need to check anymore coz we have exhausted our search
        else:
            # only one type of node can be found
            assert sum([found_leaf_node, found_backtracked_leaf_node]) == 1, "Only one type of node can be found"
            assert last_action is not None, "Last action cannot be None"
            assert action_to_take is not None, "Action to take cannot be None"
            return action_to_take
        
    def _is_leaf_node(self, tree: ProofQTree, state: ProofState, qinfo: ProofQInfo, last_action: ProofAction) -> bool:
        # A leaf node is a node which has no children or all its children are backtracked or it has a self loop and the action is get_dfns or get_thms
        return len(tree.edges[state]) == 0 or \
            (qinfo.has_self_loop and \
             qinfo.proof_env_info.progress == ProgressState.RUNNING and \
                (last_action.action_type == ProofAction.ActionType.GET_DFNS or \
                 last_action.action_type == ProofAction.ActionType.GET_THMS or \
                 last_action.action_type == ProofAction.ActionType.NONE) )

    def _has_all_backtracked_children(self, tree: ProofQTree, state: ProofState) -> bool:
        return all([state_info.qinfo.state_type == StateType.BACKTRACKED for _, state_info in tree.edges[state].items()])