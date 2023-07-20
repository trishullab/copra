#!/usr/bin/env python3

import sys
root_dir = f"{__file__.split('src')[0]}"
if root_dir not in sys.path:
    sys.path.append(root_dir)

import typing
import os
from src.rl.proof_action import ProofAction
from src.rl.abstraction import Action, Env
from src.prompt_generator.prompter import PolicyPrompter
from src.prompt_generator.agent_grammar import CoqGPTRequestGrammar, CoqGptRequestActions, GptAgentGrammar

class CoqGptPolicyPrompter(PolicyPrompter):
    def __init__(self, main_sys_prompt_path: str, example_conv_prompt_path: str):
        assert os.path.exists(main_sys_prompt_path), f"{main_sys_prompt_path} doesn't exists"
        assert os.path.exists(example_conv_prompt_path), f"{example_conv_prompt_path} doesn't exists"
        self.agent_grammar = GptAgentGrammar(user_name="example_user", agent_name="example_assistant")
        self.coq_gpt_request_grammar = CoqGPTRequestGrammar()
        conv_messages = self.agent_grammar.get_openai_conv_messages(example_conv_prompt_path, "system")
        main_message = self.agent_grammar.get_openai_main_message(main_sys_prompt_path, "system")
        self.system_messages = [main_message] + conv_messages
        pass

    def generate_prompt(self, env: Env) -> str:
        pass

    def parse_response(self, responses: list) -> typing.List[typing.Tuple[Action, float]]:
        message_contents =  self.agent_grammar.parse_openai_messages(responses, "assistant")
        actions = []
        total = len(message_contents)
        for idx, message in enumerate(message_contents):
            coq_gpt_request = self.coq_gpt_request_grammar.get_openai_request(message)
            probability = (idx + 1) / total # For now just assume that the order of the messages is the order of the actions
            if coq_gpt_request.action == CoqGptRequestActions.GET_DFNS:
                action = ProofAction(ProofAction.ActionType.GET_DFNS)
            elif coq_gpt_request.action == CoqGptRequestActions.GET_THMS:
                action = ProofAction(ProofAction.ActionType.GET_THMS)
            elif coq_gpt_request.action == CoqGptRequestActions.RUN_TACTIC:
                action = ProofAction(ProofAction.ActionType.RUN_TACTIC, tactics=coq_gpt_request.args)
            else:
                raise Exception(f"Invalid action {coq_gpt_request.action}")
            actions.append((action, probability))
        pass