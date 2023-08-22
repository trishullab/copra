#!/usr/bin/env python3

import sys
import uuid

root_dir = f"{__file__.split('src')[0]}"
if root_dir not in sys.path:
    sys.path.append(root_dir)
import os
from src.rl.abstraction import Policy
from src.agent.gpt_guided_tree_search_policy import PromptSummary, ProofQTree, StateType, TreeSearchAction, TreeSearchActionType
from src.agent.gpt_guided_tree_search_policy import ProofQInfo, ProofQTree
from src.rl.simple_proof_env import ProofEnvInfo, ProgressState
from src.rl.proof_action import ProofAction
from src.rl.proof_state import ProofState, FailedProofState
from src.agent.gpt_guided_tree_search_policy import TreeSearchAlgorithm


class GptGuidedTreeSearchPolicy(Policy):
    def __init__(self, 
        checkpoint_dir: str, 
        checkpoint_filename: str,
        checkpoint_on_exit: bool = True):
        os.path.exists(checkpoint_dir), f"Checkpoint file {checkpoint_dir} does not exist"
        checkpoint_filename is not None, "Checkpoint filename cannot be None"
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_filename = checkpoint_filename
        self._proof_q_tree : ProofQTree = None
        self.checkpoint_on_exit = checkpoint_on_exit
        self.policy_prompter = None
        self._loaded = False
    
    def __enter__(self):
        if not self.load_from_checkpoint_if_exists():
            self._proof_q_tree = ProofQTree()
        self._loaded = True
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        assert self._loaded, "Policy was not loaded"
        if self.checkpoint_on_exit:
            self.checkpoint()
    
    def load_from_checkpoint_if_exists(self):
        checkpoint_path = os.path.join(self.checkpoint_dir, self.checkpoint_filename)
        if os.path.exists(checkpoint_path) and self._proof_q_tree is None:
            with open(checkpoint_path, 'r') as f:
                self._proof_q_tree = ProofQTree.deserialize(f.read())
            return True
        return False
    
    def checkpoint(self):
        checkpoint_path = os.path.join(self.checkpoint_dir, self.checkpoint_filename)
        self._checkpoint_in_file(checkpoint_path)

    def __call__(self, state: ProofState) -> ProofAction:
        return action

    def update(self, state: ProofState, action: ProofAction, next_state: ProofState, reward: float, done: bool, info: ProofEnvInfo):
        if not done:
            # No need to update if the proof is done
            self._tree_search_algorithm.update_new_node(self._proof_q_tree, state, action, next_state, reward, done, info)

    def clone(self) -> 'GptGuidedTreeSearchPolicy':
        guid = str(uuid.uuid4())
        checkpoint_filename_without_ext, ext = os.path.splitext(self.checkpoint_filename)
        checkpoint_filename = f"{checkpoint_filename_without_ext}-{guid}.{ext}"
        self._checkpoint_in_file(os.path.join(self.checkpoint_dir, checkpoint_filename))
        copy_obj = GptGuidedTreeSearchPolicy(self.checkpoint_dir, checkpoint_filename)
        return copy_obj

    def _checkpoint_in_file(self, checkpoint_path: str):
        os.path.exists(checkpoint_path), f"Checkpoint file {checkpoint_path} does not exist"
        with open(checkpoint_path, 'w') as f:
            f.write(self._proof_q_tree.serialize())