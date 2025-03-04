#!/usr/bin/env python3

import typing
from copra.agent.dfs_policy_prompter import DfsCoqGptPolicyPrompter
from copra.prompt_generator.dfs_gpt_response_grammar import CoqGptResponse
from copra.tools.informal_proof_repo import InformalProofRepo
from itp_interface.rl.proof_action import ProofAction

class HammerDfsIsabelleGptPolicyPrompter(DfsCoqGptPolicyPrompter):
    sledgehammer_command = "show ?thesis sledgehammer"
    def __init__(self, 
            main_sys_prompt_path: str, 
            example_conv_prompt_path: str,
            num_sequences: int = 1,
            temperature: float = 0.25,
            max_tokens_per_action: int = 50,
            max_history_messages: int = 0, # This means keep no history of messages
            gpt_model_name: str = "gpt-3.5-turbo",
            secret_filepath: str = ".secrets/openai_key.json",
            k : typing.Optional[int] = None,
            retrieve_prompt_examples: bool = True,
            num_goal_per_prompt: typing.Optional[int] = None,
            training_data_path: typing.Optional[str] = None,
            metadata_filename: typing.Optional[str] = None,
            language: ProofAction.Language = ProofAction.Language.COQ,
            logger = None,
            informal_proof_repo: typing.Optional[InformalProofRepo] = None,
            lemma_name: typing.Optional[str] = None):
        super().__init__(
            main_sys_prompt_path, 
            example_conv_prompt_path, 
            num_sequences, 
            temperature, 
            max_tokens_per_action, 
            max_history_messages, 
            gpt_model_name, 
            secret_filepath, 
            k, 
            retrieve_prompt_examples, 
            num_goal_per_prompt, 
            training_data_path, 
            metadata_filename, 
            language, 
            logger, 
            informal_proof_repo, 
            lemma_name)
        self.call_num = 0
        pass

    def run_prompt(self, request: CoqGptResponse) -> list:
        # Check if sledgehammer was previously called and failed
        if (len(request.incorrect_steps) > 0 and any([step == HammerDfsIsabelleGptPolicyPrompter.sledgehammer_command for step in request.incorrect_steps])) or \
            (request.last_step == HammerDfsIsabelleGptPolicyPrompter.sledgehammer_command and request.error_message is not None):
            # If sledgehammer failed, then simply skip trying sledgehammer this time and the parent policy about the next step
            self.call_num = 1
        if self.call_num % 2 == 1:
            self.call_num += 1
            self.call_num %= 2
            return super().run_prompt(request)
        else:
            # Always try the sledgehammer first before doing anything else
            self.call_num += 1
            self.call_num %= 2
            message_content = f"[RUN TACTIC]\n{HammerDfsIsabelleGptPolicyPrompter.sledgehammer_command}\n"
            message = self.agent_grammar.get_openai_main_message_from_string(message_content, "assistant")
            message['finish_reason'] = 'stop'
            return [message]