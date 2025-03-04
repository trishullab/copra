#!/usr/bin/env python3

import typing
from collections import deque
from dataclasses import dataclass, field
from dataclasses_json import dataclass_json
from copra.rl.q_tree import QTreeStateInfo
from itp_interface.tools.dynamic_coq_proof_exec import DynamicProofExecutor as DynamicCoqProofExecutor
from itp_interface.tools.dynamic_lean_proof_exec import DynamicProofExecutor as DynamicLeanProofExecutor
from itp_interface.tools.dynamic_isabelle_proof_exec import DynamicProofExecutor as DynamicIsabelleProofExecutor
from copra.agent.gpt_guided_tree_search_policy import PromptSummary, ProofQTree, StateType, TreeSearchAction, TreeSearchActionType
from copra.agent.gpt_guided_tree_search_policy import ProofQInfo, ProofQTree
from itp_interface.rl.simple_proof_env import ProofEnvInfo, ProgressState
from itp_interface.rl.proof_action import ProofAction
from itp_interface.rl.proof_state import ProofState, FailedCoqProofState, FailedLeanProofState, FailedIsabelleProofState, FailedLean4ProofState
from copra.agent.gpt_guided_tree_search_policy import TreeSearchAlgorithm

@dataclass_json
@dataclass
class StateActionPair(object):
    state : ProofState
    action : ProofAction

    def __eq__(self, __value: object) -> bool:
        return isinstance(__value, StateActionPair) and self.state == __value.state and self.action == __value.action
    
    def __hash__(self) -> int:
        return hash((self.state, self.action))
    
    def __ge__(self, __o: object) -> bool:
        assert isinstance(__o, StateActionPair)
        assert isinstance(self.state, ProofState)
        assert isinstance(self.action, ProofAction)
        if self.state == __o.state:
            return self.action >= __o.action
        else:
            return self.state >= __o.state 
    
    def __le__(self, __o: object) -> bool:
        assert isinstance(__o, StateActionPair)
        assert isinstance(self.state, ProofState)
        assert isinstance(self.action, ProofAction)
        if self.state == __o.state:
            return self.action <= __o.action
        else:
            return self.state <= __o.state
    
    def __lt__(self, __o: object) -> bool:
        assert isinstance(__o, StateActionPair)
        assert isinstance(self.state, ProofState)
        assert isinstance(self.action, ProofAction)
        if self.state == __o.state:
            return self.action < __o.action
        else:
            return self.state < __o.state
    
    def __gt__(self, __o: object) -> bool:
        assert isinstance(__o, StateActionPair)
        assert isinstance(self.state, ProofState)
        assert isinstance(self.action, ProofAction)
        if self.state == __o.state:
            return self.action > __o.action
        else:
            return self.state > __o.state


@dataclass_json
@dataclass
class DFSTreeNode(object):
    state_action_pair: StateActionPair
    next_state_action_pair: StateActionPair
    action : ProofAction
    info : ProofEnvInfo
    reward : float
    done : bool
    incorrect_actions: typing.List[ProofAction] = field(default_factory=list)
    actions_till_now: typing.List[ProofAction] = field(default_factory=list)

class DFSTreeSearch(TreeSearchAlgorithm):
    def __init__(self, language: ProofAction.Language = ProofAction.Language.COQ):
        self._action_queue : deque = deque()
        self._search_stack : typing.List[DFSTreeNode] = []
        self._num_nodes_visited = 0
        self._bad_state_action_map : typing.Dict[ProofState, typing.Set[ProofAction]] = {}
        self.language = language
        if language == ProofAction.Language.COQ:
            self.failed_proof_state = FailedCoqProofState
        elif language == ProofAction.Language.LEAN:
            self.failed_proof_state = FailedLeanProofState
        elif language == ProofAction.Language.LEAN4:
            self.failed_proof_state = FailedLean4ProofState
        elif language == ProofAction.Language.ISABELLE:
            self.failed_proof_state = FailedIsabelleProofState
        else:
            raise NotImplementedError(f"language {self.language} not supported")
        self.has_qed = False
        pass

    def reset(self):
        self._action_queue.clear()
        self._search_stack.clear()

    def update_new_node(self, tree: ProofQTree, state: ProofState, action: ProofAction, next_state: ProofState, reward: float, done: bool, info: ProofEnvInfo):
        assert action.action_type in [ProofAction.ActionType.RUN_TACTIC, ProofAction.ActionType.GET_DFNS_THMS], "The action type should be either RUN_TACTIC, GET_DFNS or GET_THMS"
        if self.has_qed:
            self._action_queue.append(TreeSearchAction(TreeSearchActionType.STOP, state, summary=None))
            return
        if len(self._search_stack) > 0:
            last_node = self._search_stack[-1]
        else:
            last_node = None
        if self.language == ProofAction.Language.COQ:
            description_match = DynamicCoqProofExecutor.ProofFinishedDescription
            qed_tac = ["Qed."]
        elif self.language == ProofAction.Language.LEAN:
            description_match = DynamicLeanProofExecutor.ProofFinishedDescription
            qed_tac = ["end"]
        elif self.language == ProofAction.Language.ISABELLE:
            description_match = DynamicIsabelleProofExecutor.ProofFinishedDescription
            qed_tac = ["qed"]
        elif self.language == ProofAction.Language.LEAN4:
            description_match = DynamicLeanProofExecutor.ProofFinishedDescription
            qed_tac = ["\n"]
        else:
            raise NotImplementedError(f"language {self.language} not supported")
        if next_state.training_data_format is not None and next_state.training_data_format.goal_description == description_match:
            self._action_queue.append(TreeSearchAction(TreeSearchActionType.RUN_ACTION, next_state, tactics=qed_tac))
            self.has_qed = True
            return
        non_simplifying_action_message = "The proof-step does NOT simplify the goal. Try stepping back with different proof-step."
        subsequent_failed_action_message = "The proof-step ultimately leads to goals which eventually don't simplify. Try stepping back with a different proof-step."
        current_state_action_pair = StateActionPair(state, ProofAction(ProofAction.ActionType.NONE, self.language))
        next_state_action_pair = StateActionPair(next_state, action)
        new_node = DFSTreeNode(current_state_action_pair, next_state_action_pair, action, info, reward, done)
        current_node_is_correct = True
        bad_action_state = state in self._bad_state_action_map and action in self._bad_state_action_map[state]
        if new_node.info.progress == ProgressState.FAILED:
            assert new_node.info.progress == ProgressState.FAILED, "The progress should be FAILED"
            new_node.next_state_action_pair.state = self.failed_proof_state # This is to ensure that there are no cycles in the tree
            current_node_is_correct = False
        elif self._check_if_state_is_harder(current_state_action_pair, next_state_action_pair):
            if action.action_type == ProofAction.ActionType.RUN_TACTIC:
                # Backtrack to the previous state because we ran a tactic which did not simplify the goal
                self._action_queue.append(TreeSearchAction(TreeSearchActionType.BACKTRACK, state, summary=None))
            new_node.info.progress = ProgressState.FAILED
            new_node.info.error_message = non_simplifying_action_message
            new_node.next_state_action_pair.state = self.failed_proof_state # This is to ensure that there are no cycles in the tree
            current_node_is_correct = False
        else:
            assert new_node.info.progress == ProgressState.STATE_CHANGED or new_node.info.progress == ProgressState.STATE_UNCHANGED or new_node.info.progress == ProgressState.DONE, "The progress should be either STATE_CHANGED or STATE_UNCHANGED"
            assert not new_node.state_action_pair <= new_node.next_state_action_pair, "The next state should not be harder than the current state"
            current_node_is_correct = True

        if new_node.state_action_pair.state in self._bad_state_action_map:
            # include the incorrect actions discovered for the same state in the past
            new_node.incorrect_actions = list(self._bad_state_action_map[new_node.state_action_pair.state])
            # sort the incorrect actions by name
            new_node.incorrect_actions.sort(key=lambda x: x.name)
        
        if bad_action_state:
            # We know that the action is a repition of a bad action, so we should not run it again
            if current_node_is_correct:
                # We must undo the current action because it is a repetition of a bad action which eventually leads to a failed state
                self._action_queue.append(TreeSearchAction(TreeSearchActionType.BACKTRACK, state, summary=None))
            # We know that last_node is no longer useful, so we pop it from the stack
            if last_node is None:
                # There is nothing in the queue the search is over
                self._action_queue.append(TreeSearchAction(TreeSearchActionType.STOP, state, summary=None))
            else:
                last_node = self._search_stack.pop()
                # Check if the last node is a failed node
                if last_node.next_state_action_pair.state == self.failed_proof_state:
                    # If the last node is a failed node, we should pop the next node
                    last_node = self._search_stack[-1] if len(self._search_stack) > 0 else None
                if last_node is None:
                    # There is nothing in the queue the search is over
                    self._action_queue.append(TreeSearchAction(TreeSearchActionType.STOP, state, summary=None))
                else:
                    # This node should not be a failed node
                    assert last_node.next_state_action_pair.state != self.failed_proof_state, "The last node's next state should not be self.failed_proof_state"
                    if last_node.action.action_type == ProofAction.ActionType.RUN_TACTIC:
                        # Add backtracking if the last action was a tactic
                        self._action_queue.append(TreeSearchAction(TreeSearchActionType.BACKTRACK, state, summary=None))
                    # Deem the last action as invalid
                    last_node.next_state_action_pair.state = self.failed_proof_state
                    last_node.info.progress = ProgressState.FAILED
                    last_node.info.error_message = subsequent_failed_action_message
                    # Add the action to failed state
                    if last_node.state_action_pair.state not in self._bad_state_action_map:
                        self._bad_state_action_map[last_node.state_action_pair.state] = set()
                    self._bad_state_action_map[last_node.state_action_pair.state].add(last_node.action)
        elif last_node is None or last_node.next_state_action_pair.state != self.failed_proof_state:
            if last_node is not None:
                new_node.actions_till_now = last_node.actions_till_now + [last_node.action]
            if last_node is None and bad_action_state:
                self._action_queue.append(TreeSearchAction(TreeSearchActionType.STOP, state, summary=None))
            else:
                self._search_stack.append(new_node)
            if new_node.info.progress == ProgressState.FAILED:
                if state not in self._bad_state_action_map:
                    self._bad_state_action_map[state] = set()
                self._bad_state_action_map[state].add(action)
        elif current_node_is_correct:
            assert last_node.next_state_action_pair.state == self.failed_proof_state, "The last node's next state should be self.failed_proof_state"
            assert last_node.state_action_pair.state == new_node.state_action_pair.state, "There cannot be a jump in the states"
            # Pop the failed node from the stack
            self._search_stack.pop()
            if action == last_node.next_state_action_pair.action or action in last_node.incorrect_actions or bad_action_state:
                if bad_action_state and action not in last_node.incorrect_actions:
                    last_node.incorrect_actions.append(action)
                last_node = self._search_stack[-1] if len(self._search_stack) > 0 else None
                if action.action_type == ProofAction.ActionType.RUN_TACTIC:
                    # Add backtracking if the last action was a tactic
                    self._action_queue.append(TreeSearchAction(TreeSearchActionType.BACKTRACK, state, summary=None))
                # This is the case when we have backtracked to the previous state and we are trying to run the same tactic again
                # Since we should not have run the same tactic again, we should just bactrack the new run.
                if last_node is None:
                    # We are done searching because it repeated the same wrong action, even after warning
                    self._action_queue.append(TreeSearchAction(TreeSearchActionType.STOP, state, summary=None))
                else:
                    assert last_node.next_state_action_pair.state != self.failed_proof_state, "The last node's next state should not be self.failed_proof_state"
                    if last_node.action.action_type == ProofAction.ActionType.RUN_TACTIC:
                        self._action_queue.append(TreeSearchAction(TreeSearchActionType.BACKTRACK, state, summary=None))
                    # Deem the last action as invalid
                    last_node.next_state_action_pair.state = self.failed_proof_state
                    last_node.info.progress = ProgressState.FAILED
                    last_node.info.error_message = subsequent_failed_action_message
                    # # Add the action to failed state
                    # if last_node.action not in last_node.incorrect_actions:
                    #     last_node.incorrect_actions.append(last_node.action)
                    if last_node.state_action_pair.state not in self._bad_state_action_map:
                        self._bad_state_action_map[last_node.state_action_pair.state] = set()
                    self._bad_state_action_map[last_node.state_action_pair.state].add(last_node.action)
            else:
                # Update the last node as the older node is popped
                last_node = self._search_stack[-1] if len(self._search_stack) > 0 else None
                if last_node is not None:
                    new_node.actions_till_now = last_node.actions_till_now + [last_node.action]
                # Add the new node to the stack
                self._search_stack.append(new_node)
                if new_node.info.progress == ProgressState.FAILED:
                    if state not in self._bad_state_action_map:
                        self._bad_state_action_map[state] = set()
                    self._bad_state_action_map[state].add(action)
        else:
            assert last_node.state_action_pair.state == new_node.state_action_pair.state, "There cannot be a jump in the states"
            assert last_node.next_state_action_pair.state == self.failed_proof_state, "The last node's next state should be self.failed_proof_state"
            if action in last_node.incorrect_actions or new_node.action == last_node.action or bad_action_state:
                if bad_action_state and action not in last_node.incorrect_actions:
                    last_node.incorrect_actions.append(action)
                # Pop from the stack, because we no longer want to use this action again
                self._search_stack.pop()
                # Update the last node as the older node is popped
                last_node = self._search_stack[-1] if len(self._search_stack) > 0 else None
                if last_node is None:
                    # There is nothing in the queue the search is over
                    self._action_queue.append(TreeSearchAction(TreeSearchActionType.STOP, state, summary=None))
                else:
                    assert last_node.next_state_action_pair.state != self.failed_proof_state, "The last node's next state should not be self.failed_proof_state"
                    if last_node.action.action_type == ProofAction.ActionType.RUN_TACTIC:
                        # Add backtracking if the last action was a tactic
                        self._action_queue.append(TreeSearchAction(TreeSearchActionType.BACKTRACK, state, summary=None))
                    # Deem the last action as invalid
                    last_node.next_state_action_pair.state = self.failed_proof_state
                    last_node.info.progress = ProgressState.FAILED
                    last_node.info.error_message = subsequent_failed_action_message
                    # # Add the action to failed state
                    # if action not in last_node.incorrect_actions:
                    #     last_node.incorrect_actions.append(action)
                    if last_node.state_action_pair.state not in self._bad_state_action_map:
                        self._bad_state_action_map[last_node.state_action_pair.state] = set()
                    self._bad_state_action_map[last_node.state_action_pair.state].add(last_node.action)
            else:
                last_node.incorrect_actions.append(action)
                # sort the incorrect actions by name
                last_node.incorrect_actions.sort(key=lambda x: x.name)
                # Update the incorrect actions in the bad state action map
                if last_node.state_action_pair.state not in self._bad_state_action_map:
                    self._bad_state_action_map[last_node.state_action_pair.state] = set()
                self._bad_state_action_map[last_node.state_action_pair.state].add(last_node.action)
                if state not in self._bad_state_action_map:
                    self._bad_state_action_map[state] = set()
                self._bad_state_action_map[state].add(action)
                last_node.action = new_node.action
                last_node.next_state_action_pair.action = new_node.next_state_action_pair.action
                last_node.next_state_action_pair.state = self.failed_proof_state
                last_node.info = new_node.info
    
    def estimate_q_value(self, tree: ProofQTree, state: ProofState, action: ProofAction, next_state: ProofState, reward: float, done: bool, info: ProofEnvInfo) -> float:
        return super().estimate_q_value(tree, state, action, next_state, reward, done, info)
    
    def __call__(self, tree: ProofQTree, state: ProofState) -> TreeSearchAction:
        if len(self._action_queue) > 0:
            return self._action_queue.popleft()
        elif len(self._search_stack) == 0:
            qtree_state_info = QTreeStateInfo(state, 
                ProofQInfo(0.0, False, 0.0, has_loop=False, distance_from_root=0, proof_env_info=None, state_type=StateType.UNDISCOVERED))
            incorrect_actions = []
            if state in self._bad_state_action_map:
                # This will be the case when proof search backtracks to the root state, but we shouldn't forget the incorrect actions
                incorrect_actions = list(self._bad_state_action_map[state])
                incorrect_actions.sort(key=lambda x: x.name)
            # There are no nodes in the tree, so we have to just give the summary from the proof state.
            return TreeSearchAction(TreeSearchActionType.NEXT_ACTION_SUMMARY_PROMPT, state, 
                summary=PromptSummary(incorrect_actions, [], None, qtree_state_info))
        else:
            return self._dfs(tree, state)
    
    def _dfs(self, tree: ProofQTree, state: ProofState) -> TreeSearchAction:
        assert len(self._search_stack) > 0, "The search stack should not be empty"
        last_node = self._search_stack[-1]
        distance_from_root = len(self._search_stack)
        incorrect_actions_set = set()
        if state in self._bad_state_action_map:
            for action in self._bad_state_action_map[state]:
                incorrect_actions_set.add(action)
        if last_node.next_state_action_pair.state == self.failed_proof_state:
            assert last_node.state_action_pair.state == state, "The last node's current state should be the current state"
            assert last_node.info.progress == ProgressState.FAILED, "The last node's progress should be FAILED"
            assert last_node.info.error_message is not None, "The last node's error message should not be None"
            # sort the incorrect actions by name
            # Make sure that incorrect actions are updated
            for action in last_node.incorrect_actions:
                incorrect_actions_set.add(action)
            incorrect_actions = list(incorrect_actions_set)
            incorrect_actions.sort(key=lambda x: x.name)
            self._num_nodes_visited += 1
            return TreeSearchAction(TreeSearchActionType.FAILED_ACTION_SUMMARY_PROMPT,
                    last_node.state_action_pair.state, summary = PromptSummary(
                        incorrect_actions,
                        last_node.actions_till_now,
                        last_node.action,
                        QTreeStateInfo(last_node.state_action_pair.state, 
                            ProofQInfo(
                                last_node.reward, 
                                last_node.done, 
                                qval = -1.0 * distance_from_root, 
                                distance_from_root = distance_from_root,
                                proof_env_info=last_node.info))))
        elif last_node.next_state_action_pair.state == state:
            assert last_node.info.progress != ProgressState.FAILED, "The last node's progress should not be FAILED"
            assert last_node.info.error_message is None, "The last node's error message should be None"
            self._num_nodes_visited += 1
            incorrect_actions = list(incorrect_actions_set)
            incorrect_actions.sort(key=lambda x: x.name)
            return TreeSearchAction(TreeSearchActionType.NEXT_ACTION_SUMMARY_PROMPT,
                    last_node.next_state_action_pair.state, summary = PromptSummary(
                        incorrect_actions,
                        last_node.actions_till_now,
                        last_node.action,
                        QTreeStateInfo(last_node.next_state_action_pair.state, 
                            ProofQInfo(
                                last_node.reward, 
                                last_node.done, 
                                qval = -1.0 * distance_from_root, 
                                distance_from_root = distance_from_root,
                                proof_env_info=last_node.info))))
        else:
            raise Exception("The last node's next state should either be the current state or a failed state")

    def _check_if_state_is_harder(self, current_state_action_pair: StateActionPair, next_state_action_pair: StateActionPair) -> bool:
        if current_state_action_pair <= next_state_action_pair:
            return True
        else:
            for node in self._search_stack:
                if node.next_state_action_pair <= next_state_action_pair:
                    return True
            return False