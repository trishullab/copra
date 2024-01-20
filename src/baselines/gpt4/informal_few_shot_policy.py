#!/usr/bin/env python3

import sys

root_dir = f"{__file__.split('src')[0]}"
if root_dir not in sys.path:
    sys.path.append(root_dir)
import json
import os
import uuid
import typing
import logging
from src.agent.gpt_guided_tree_search_policy import ProofQTree
from src.agent.gpt_guided_tree_search_policy import ProofQTree
from src.agent.rate_limiter import InvalidActionException
from src.baselines.gpt4.informal_few_shot_grammar import InformalFewShotGptResponse
from src.baselines.gpt4.informal_few_shot_policy_prompter import InformalFewShotGptPolicyPrompter
from src.rl.abstraction import Policy
from src.rl.proof_action import ProofAction
from src.rl.proof_state import ProofState
from src.rl.simple_proof_env import ProofEnvInfo
from src.tools.informal_proof_repo import InformalProofRepo


class InformalFewShotGptPolicy(Policy):
    def __init__(self,
        lemma_name: str, 
        checkpoint_dir: str, 
        checkpoint_filename: str,
        policy_prompter: InformalFewShotGptPolicyPrompter,
        informal_proof_repo: typing.Optional[InformalProofRepo],
        checkpoint_on_exit: bool = True,
        language: ProofAction.Language = ProofAction.Language.LEAN,
        logger: logging.Logger = None,
        informal_proof_dump_dir: str = None):
        os.path.exists(checkpoint_dir), f"Checkpoint file {checkpoint_dir} does not exist"
        assert checkpoint_filename is not None, "Checkpoint filename cannot be None"
        assert policy_prompter is not None, "Policy prompter cannot be None"
        assert informal_proof_repo is not None, "Informal proof repo cannot be None"
        assert language == ProofAction.Language.LEAN or language == ProofAction.Language.ISABELLE,
            "Only Lean or Isabelle is supported for informal proofs"
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_filename = checkpoint_filename
        self._policy_prompter = policy_prompter
        self._proof_q_tree : ProofQTree = None
        self.checkpoint_on_exit = checkpoint_on_exit
        self.policy_prompter = None
        self._loaded = False
        self.logger = logger if logger is not None else logging.getLogger(__name__)
        self._asked_for_dfns_and_lms = False
        self._asked_for_proof = False
        self._num_api_calls = 0
        self.language = language
        self.lemma_name = lemma_name
        self.informal_proof_repo = informal_proof_repo
        self.informal_proof_dump_dir = informal_proof_dump_dir
    
    def __enter__(self):
        if not self.load_from_checkpoint_if_exists():
            self._proof_q_tree = ProofQTree()
        self._policy_prompter.__enter__()
        self._loaded = True
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        assert self._loaded, "Policy was not loaded"
        if self.checkpoint_on_exit:
            self.checkpoint()
        self._policy_prompter.__exit__(exc_type, exc_value, traceback)
    
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
        # assert len(state.training_data_format.start_goals) == 1, "At the begining of the proof there is exactly one goal"
        # There can actually be more than one goals at the beginning of the proof
        if not self._asked_for_dfns_and_lms:
            if self.language == ProofAction.Language.COQ:
                if len(state.training_data_format.all_useful_defns_theorems) == 0:
                    self._asked_for_dfns_and_lms = True
                    return ProofAction(ProofAction.ActionType.GET_DFNS_THMS, self.language)
            elif self.language == ProofAction.Language.LEAN:
                self._asked_for_dfns_and_lms = True
                # Move on because we don't support retrieving definitions and theorems for Lean as of now
            elif self.language == ProofAction.Language.ISABELLE:
                if len(state.training_data_format.all_useful_defns_theorems) == 0:
                    self._asked_for_dfns_and_lms = True
                    return ProofAction(ProofAction.ActionType.GET_DFNS_THMS, self.language)
        if not self._asked_for_proof:
            success = False
            tries = 10
            exceptions = []
            if self.language == ProofAction.Language.LEAN or self.language == ProofAction.Language.ISABELLE:
                theorem_stmt, _ = self.informal_proof_repo.get_informal_thm_proof(self.lemma_name)
                gpt_response = InformalFewShotGptResponse(theorem=theorem_stmt)
            else:
                raise Exception(f"Unsupported language {self.language}")
            while not success and tries > 0:
                try:
                    responses = self._policy_prompter.run_prompt(gpt_response)
                    actions_tuple : typing.List[typing.Tuple[ProofAction, float]] = self._policy_prompter.parse_response(responses)
                    chosen_message = actions_tuple[0][0].original_message # Selecting only top action here
                    self.logger.info(f"Chosen message: \n\n{chosen_message['content']}")
                    # The proofs will not be added to history
                    success = True
                except InvalidActionException as e:
                    self.logger.error("Got an exception while trying to parse response generated by GPT")
                    self.logger.exception(e)
                tries -= 1
            self._asked_for_proof = True # We only ask for proof once because it is not an interactive process like an agent
            if not success:
                raise Exception(f"Failed to get valid action after {tries} tries. Exceptions:\n {exceptions}")
            action = actions_tuple[0][0]
            assert action.action_type == ProofAction.ActionType.INFORMAL, "Only proof action is supported for informal proofs"
            if self.informal_proof_dump_dir is not None:
                proof_json = {
                    "problem_name": self.lemma_name,
                    "informal_statement": theorem_stmt,
                    "informal_proof": action.kwargs["proof"],
                }
                with open(os.path.join(self.informal_proof_dump_dir, f"{self.lemma_name}.json"), "w") as f:
                    f.write(json.dumps(proof_json))
            return action
        else:
            return ProofAction(ProofAction.ActionType.EXIT, self.language)

    def update(self, state: ProofState, action: ProofAction, next_state: ProofState, reward: float, done: bool, info: ProofEnvInfo):
        pass

    def clone(self) -> 'InformalFewShotGptPolicy': 
        guid = str(uuid.uuid4())
        checkpoint_filename_without_ext, ext = os.path.splitext(self.checkpoint_filename)
        checkpoint_filename = f"{checkpoint_filename_without_ext}-{guid}.{ext}"
        self._checkpoint_in_file(os.path.join(self.checkpoint_dir, checkpoint_filename))
        copy_obj = InformalFewShotGptPolicy(checkpoint_filename, self.checkpoint_dir, checkpoint_filename)
        return copy_obj

    def _checkpoint_in_file(self, checkpoint_path: str):
        os.path.exists(checkpoint_path), f"Checkpoint file {checkpoint_path} does not exist"
        with open(checkpoint_path, 'w') as f:
            f.write(self._proof_q_tree.serialize())
    
    def get_efficiency_info(self) -> typing.Dict[str, typing.Any]:
        return {
            "queries": self._policy_prompter.get_efficiency_info()["api_calls"],
        }