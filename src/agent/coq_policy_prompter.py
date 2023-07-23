#!/usr/bin/env python3

import sys

root_dir = f"{__file__.split('src')[0]}"
if root_dir not in sys.path:
    sys.path.append(root_dir)

import typing
import os
import time
from src.gpts.gpt_access import GptAccess
from src.rl.proof_action import ProofAction
from src.prompt_generator.prompter import PolicyPrompter
from src.prompt_generator.agent_grammar import CoqGPTRequestGrammar, CoqGPTResponseGrammar, CoqGptRequestActions, CoqGptResponse, GptAgentGrammar

class RateLimiter(object):
    def __init__(self, token_limit_per_min: int, request_limit_per_min: int):
        assert token_limit_per_min > 0, "Token limit must be greater than 0"
        assert request_limit_per_min > 0, "Request limit must be greater than 0"
        self.token_limit_per_min = token_limit_per_min
        self.request_limit_per_min = request_limit_per_min
        self._token_count = 0
        self._request_count = 0
        self._last_request_time = None
    
    def check(self, new_tokens: int = 0) -> bool:
        current_time = time.time()
        if self._last_request_time is None:
            self._last_request_time = current_time
        if current_time - self._last_request_time <= 60:
            if (self._token_count + new_tokens) >= self.token_limit_per_min or \
            (self._request_count + 1) >= self.request_limit_per_min:
                return False
        return True
    
    def update(self, token_count: int, request_start_time: float, request_end_time: float):
        self._token_count += token_count
        self._request_count += 1
        self._last_request_time = (request_start_time + request_end_time) / 2

class InvalidActionException(Exception):
    def __init__(self, message):
        self.message = message
    pass

class CoqGptPolicyPrompter(PolicyPrompter):
    def __init__(self, 
            main_sys_prompt_path: str, 
            example_conv_prompt_path: str,
            num_sequences: int = 1,
            temperature: float = 0.25,
            max_tokens_per_action: int = 50,
            gpt_model_name: str = "gpt-3.5-turbo",
            secret_filepath: str = ".secrets/openai_key.json"):
        assert os.path.exists(main_sys_prompt_path), f"{main_sys_prompt_path} doesn't exists"
        assert os.path.exists(example_conv_prompt_path), f"{example_conv_prompt_path} doesn't exists"
        self.agent_grammar = GptAgentGrammar(user_name="example_user", agent_name="example_assistant")
        self.coq_gpt_request_grammar = CoqGPTRequestGrammar()
        self.coq_gpt_response_grammar = CoqGPTResponseGrammar()
        conv_messages = self.agent_grammar.get_openai_conv_messages(example_conv_prompt_path, "system")
        main_message = self.agent_grammar.get_openai_main_message(main_sys_prompt_path, "system")
        self.system_messages = [main_message] + conv_messages
        self._gpt_access = GptAccess(secret_filepath=secret_filepath, model_name=gpt_model_name)
        self._token_limit_per_min = GptAccess.gpt_model_info[gpt_model_name]["token_limit_per_min"]
        self._request_limit_per_min = GptAccess.gpt_model_info[gpt_model_name]["request_limit_per_min"]
        self._max_token_per_prompt = GptAccess.gpt_model_info[gpt_model_name]["max_token_per_prompt"]
        self._rate_limiter = RateLimiter(self._token_limit_per_min, self._request_limit_per_min)
        self.temperature = temperature
        self.num_sequences = num_sequences
        self.system_token_count = self._gpt_access.num_tokens_from_messages(self.system_messages)
        self._max_tokens_per_action = max_tokens_per_action
        self._history_token_count = 0
        self._message_history = []
        self._message_history_token_count = []
        pass

    def add_to_history(self, message: typing.Any):
        message_token_count = self._gpt_access.num_tokens_from_messages([message])
        self._message_history.append(message)
        self._message_history_token_count.append(message_token_count)
        self._history_token_count += message_token_count

    def run_prompt(self, request: CoqGptResponse) -> list:
        prompt_message = self.coq_gpt_response_grammar.format_as_per_grammar(request)
        prompt_message = self.agent_grammar.get_openai_main_message_from_string(prompt_message, "user")
        prompt_messages = [prompt_message]
        prompt_token_count = self._gpt_access.num_tokens_from_messages(prompt_messages)
        total_token_count = self.system_token_count + self._history_token_count + prompt_token_count
        history_idx = 0
        while total_token_count > self._max_token_per_prompt:
            self._history_token_count -= self._message_history_token_count[history_idx]
            total_token_count = self.system_token_count + self._history_token_count + prompt_token_count
            history_idx += 1
        self._message_history = self._message_history[history_idx:]
        self._message_history_token_count = self._message_history_token_count[history_idx:]
        self._message_history.append(prompt_message)
        self._message_history_token_count.append(prompt_token_count)
        messages = self.system_messages + self._message_history
        while not self._rate_limiter.check(total_token_count):
            current_time = time.time()
            time_to_sleep = min(1, 60 - (current_time - self._rate_limiter._last_request_time))
            time.sleep(time_to_sleep)
        success = False
        tries = 10
        time_to_sleep = 60
        exp_factor = 1.25
        while not success and tries > 0:
            try:
                request_start_time = time.time()
                responses, usage = self._gpt_access.complete_chat(
                    messages,
                    n=self.num_sequences,
                    temperature=self.temperature,
                    max_tokens=self._max_tokens_per_action,
                    stop=["[END]"])
                request_end_time = time.time()
                success = True
            except Exception as e:
                time.sleep(time_to_sleep)
                print(e)
                responses = []
                usage = {}
                time_to_sleep *= exp_factor # Exponential backoff
            tries -= 1
        if not success:
            raise Exception(f"Failed to get valid response after {tries} tries")
        self._rate_limiter.update(usage["total_tokens"], request_start_time, request_end_time)
        return responses

    def parse_response(self, responses: list) -> typing.List[typing.Tuple[typing.Any, ProofAction, float]]:
        message_contents =  self.agent_grammar.parse_openai_messages(responses, "assistant")
        actions = []
        total = len(message_contents)
        for idx, message in enumerate(message_contents):
            try:
                coq_gpt_request = self.coq_gpt_request_grammar.get_openai_request(message)
            except Exception as e:
                error_message = f"Invalid response:\n {str(e)}"
                raise InvalidActionException(error_message)
            probability = (idx + 1) / total # For now just assume that the order of the messages is the order of the actions
            if coq_gpt_request.action == CoqGptRequestActions.GET_DFNS:
                action = ProofAction(ProofAction.ActionType.GET_DFNS)
            elif coq_gpt_request.action == CoqGptRequestActions.GET_THMS:
                action = ProofAction(ProofAction.ActionType.GET_THMS)
            elif coq_gpt_request.action == CoqGptRequestActions.RUN_TACTIC:
                action = ProofAction(ProofAction.ActionType.RUN_TACTIC, tactics=coq_gpt_request.args)
            else:
                raise Exception(f"Invalid action {coq_gpt_request.action}")
            actions.append((responses[idx], action, probability))
        return actions

if __name__ == "__main__":
    prompter = CoqGptPolicyPrompter(
        main_sys_prompt_path="data/prompts/system/coq-proof-agent-role.md",
        example_conv_prompt_path="data/prompts/conversation/coq-proof-agent-example.md")
    request = CoqGPTResponseGrammar(
        action=CoqGptRequestActions.RUN_TACTIC,
        args=["reflexivity"])
    responses = prompter.run_prompt(request)
    print(responses)
    actions = prompter.parse_response(responses)
    print(actions)