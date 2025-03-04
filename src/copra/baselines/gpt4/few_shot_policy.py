#!/usr/bin/env python3

import os
import uuid
import typing
import logging
from copra.agent.gpt_guided_tree_search_policy import ProofQTree
from copra.agent.gpt_guided_tree_search_policy import ProofQTree
from copra.agent.rate_limiter import InvalidActionException
from copra.baselines.gpt4.few_shot_grammar import FewShotGptResponse
from copra.baselines.gpt4.few_shot_policy_prompter import FewShotGptPolicyPrompter
from itp_interface.rl.abstraction import Policy
from itp_interface.rl.proof_action import ProofAction
from itp_interface.rl.proof_state import ProofState
from itp_interface.rl.simple_proof_env import ProofEnvInfo
from copra.tools.informal_proof_repo import InformalProofRepo


class FewShotGptPolicy(Policy):
    def __init__(self,
        lemma_name: str, 
        checkpoint_dir: str, 
        checkpoint_filename: str,
        policy_prompter: FewShotGptPolicyPrompter,
        checkpoint_on_exit: bool = True,
        language: ProofAction.Language = ProofAction.Language.COQ,
        logger: logging.Logger = None,
        informal_proof_repo: typing.Optional[InformalProofRepo] = None):
        os.path.exists(checkpoint_dir), f"Checkpoint file {checkpoint_dir} does not exist"
        assert checkpoint_filename is not None, "Checkpoint filename cannot be None"
        assert policy_prompter is not None, "Policy prompter cannot be None"
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
            elif self.language == ProofAction.Language.LEAN4:
                self._asked_for_dfns_and_lms = True
                # Move on because we don't support retrieving definitions and theorems for Lean4 as of now
            elif self.language == ProofAction.Language.ISABELLE:
                if len(state.training_data_format.all_useful_defns_theorems) == 0:
                    self._asked_for_dfns_and_lms = True
                    return ProofAction(ProofAction.ActionType.GET_DFNS_THMS, self.language)
        if not self._asked_for_proof:
            success = False
            tries = 10
            exceptions = []
            if self.language == ProofAction.Language.COQ:
                gpt_response = FewShotGptResponse(
                    theorem=state.training_data_format.start_goals[0].goal,
                    defintions=[str(state.training_data_format.all_useful_defns_theorems[lemma_ref.lemma_idx]) for lemma_ref in state.training_data_format.start_goals[0].relevant_defns],
                    lemmas=[str(state.training_data_format.all_useful_defns_theorems[lemma_ref.lemma_idx]) for lemma_ref in state.training_data_format.start_goals[0].possible_useful_theorems_local], # We don't allow any sophisticated retrieval action here
                )
            elif self.language == ProofAction.Language.LEAN:
                theorem_statement_with_name = state.theorem_statement_with_name
                # Replace the theorem name with the some anonymous name
                theorem_statement_with_name = theorem_statement_with_name.replace(state.theorem_name, "some_theorem")
                gpt_response = FewShotGptResponse(
                    theorem=theorem_statement_with_name,
                    defintions=[],
                    lemmas=[],
                )
                if self.informal_proof_repo is not None:
                    informal_thm, informal_proof = self.informal_proof_repo.get_informal_thm_proof(self.lemma_name)
                    gpt_response.informal_theorem = informal_thm
                    gpt_response.informal_proof = informal_proof
            elif self.language == ProofAction.Language.LEAN4:
                theorem_statement_with_name = state.theorem_statement_with_name
                # Replace the theorem name with the some anonymous name
                theorem_statement_with_name = theorem_statement_with_name.replace(state.theorem_name, "some_theorem")
                gpt_response = FewShotGptResponse(
                    theorem=theorem_statement_with_name,
                    defintions=[],
                    lemmas=[],
                )
                if self.informal_proof_repo is not None:
                    informal_thm, informal_proof = self.informal_proof_repo.get_informal_thm_proof(self.lemma_name)
                    gpt_response.informal_theorem = informal_thm
                    gpt_response.informal_proof = informal_proof
            elif self.language == ProofAction.Language.ISABELLE:
                theorem_statement_with_name = state.theorem_statement_with_name
                # Replace the theorem name with the some anonymous name
                theorem_statement_with_name = theorem_statement_with_name.replace(state.theorem_name, "some_theorem")
                gpt_response = FewShotGptResponse(
                    theorem=theorem_statement_with_name,
                    defintions=[],
                    lemmas=[],
                )
                if self.informal_proof_repo is not None:
                    informal_thm, informal_proof = self.informal_proof_repo.get_informal_thm_proof(self.lemma_name)
                    gpt_response.informal_theorem = informal_thm
                    gpt_response.informal_proof = informal_proof
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
            return action
        else:
            return ProofAction(ProofAction.ActionType.EXIT, self.language)

    def update(self, state: ProofState, action: ProofAction, next_state: ProofState, reward: float, done: bool, info: ProofEnvInfo):
        pass

    def clone(self) -> 'FewShotGptPolicy': 
        guid = str(uuid.uuid4())
        checkpoint_filename_without_ext, ext = os.path.splitext(self.checkpoint_filename)
        checkpoint_filename = f"{checkpoint_filename_without_ext}-{guid}.{ext}"
        self._checkpoint_in_file(os.path.join(self.checkpoint_dir, checkpoint_filename))
        copy_obj = FewShotGptPolicy(checkpoint_filename, self.checkpoint_dir, checkpoint_filename)
        return copy_obj

    def _checkpoint_in_file(self, checkpoint_path: str):
        os.path.exists(checkpoint_path), f"Checkpoint file {checkpoint_path} does not exist"
        with open(checkpoint_path, 'w') as f:
            f.write(self._proof_q_tree.serialize())
    
    def get_efficiency_info(self) -> typing.Dict[str, typing.Any]:
        return {
            "queries": self._policy_prompter.get_efficiency_info()["api_calls"],
        }