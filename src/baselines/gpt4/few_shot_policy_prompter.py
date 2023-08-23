#!/usr/bin/env python3

import sys

root_dir = f"{__file__.split('src')[0]}"
if root_dir not in sys.path:
    sys.path.append(root_dir)

import typing
import os
import time
from openai.error import InvalidRequestError
import logging
from src.agent.rate_limiter import RateLimiter, InvalidActionException
from src.agent.gpt_guided_tree_search_policy import TreeSearchAction
from src.gpts.gpt_access import GptAccess
from src.rl.proof_action import ProofAction
from src.prompt_generator.prompter import PolicyPrompter
from src.prompt_generator.dfs_agent_grammar import DfsAgentGrammar
from src.baselines.gpt4.few_shot_grammar import FewShotGptKeywords, FewShotGptRequest, FewShotGptRequestGrammar, FewShotGptResponseGrammar


class FewShotGptPolicyPrompter(PolicyPrompter):
    def __init__(self, 
            main_sys_prompt_path: str, 
            example_conv_prompt_path: str,
            num_sequences: int = 1,
            temperature: float = 0.25,
            max_tokens_per_action: int = 250,
            max_history_messages: int = 0, # This means keep no history of messages
            gpt_model_name: str = "gpt-3.5-turbo",
            secret_filepath: str = ".secrets/openai_key.json",
            k : typing.Optional[int] = None,
            logger = None):
        assert os.path.exists(main_sys_prompt_path), f"{main_sys_prompt_path} doesn't exists"
        assert os.path.exists(example_conv_prompt_path), f"{example_conv_prompt_path} doesn't exists"
        self.agent_grammar = DfsAgentGrammar(user_name="example_user", agent_name="example_assistant")
        self.coq_gpt_request_grammar = FewShotGptRequestGrammar()
        self.coq_gpt_response_grammar = FewShotGptResponseGrammar()
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
        self._max_history_messages = max_history_messages
        self._k = k
        self.logger = logger if logger is not None else logging.getLogger(__name__)
        pass

    def add_to_history(self, message: typing.Any):
        message_token_count = self._gpt_access.num_tokens_from_messages([message])
        self._message_history.append(message)
        self._message_history_token_count.append(message_token_count)
        self._history_token_count += message_token_count

    def _constrain_tokens_in_history(self, prompt_message, prompt_token_count: int, max_tokens_per_action: int) -> list:
        if len(self._message_history) >= self._max_history_messages:
            history_idx = len(self._message_history) - self._max_history_messages
        else:
            history_idx = 0
        if history_idx < len(self._message_history):
            # There is no point in checking the token count if there is no history to be maintained
            total_token_count = self.system_token_count + self._history_token_count + prompt_token_count
            max_token_per_prompt = min(self._max_token_per_prompt, self._max_token_per_prompt - max_tokens_per_action)
            assert max_token_per_prompt > 0, "Max token per prompt must be greater than 0, please decrease max_tokens_per_action"
            tokens_shredded = False
            remove_cnt  = 0
            history_count = self._history_token_count
            while total_token_count >= max_token_per_prompt and history_idx < len(self._message_history):
                self.logger.warning(f"Tokens exceeded removing history at index {history_idx}: {total_token_count} >= {max_token_per_prompt}")
                history_count -= self._message_history_token_count[history_idx]
                total_token_count = self.system_token_count + history_count + prompt_token_count
                history_idx += 1
                tokens_shredded = True
                remove_cnt += 1
            if remove_cnt % 2 == 1 and history_idx < len(self._message_history):
                history_count -= self._message_history_token_count[history_idx]
                total_token_count = self.system_token_count + history_count + prompt_token_count
                history_idx += 1
            if tokens_shredded:
                self.logger.warning(f"Shredded tokens from history. New total token count: {total_token_count}, max token per prompt: {max_token_per_prompt}, history token count: {self._history_token_count}, prompt token count: {prompt_token_count}")
            if total_token_count >= max_token_per_prompt:
                self.logger.warning(f"Total token count {total_token_count} is still greater than max token per prompt {max_token_per_prompt}.")
        else:
            total_token_count = self.system_token_count + prompt_token_count
        if history_idx > 0:
            for idx in range(min(history_idx, len(self._message_history))):
                self._history_token_count -= self._message_history_token_count[idx]
        self._message_history = self._message_history[history_idx:]
        self._message_history_token_count = self._message_history_token_count[history_idx:]
        self._message_history.append(prompt_message)
        self._message_history_token_count.append(prompt_token_count)
        self._history_token_count += prompt_token_count
        messages = self.system_messages + self._message_history
        return messages, total_token_count
    
    def _throttle_if_needed(self, total_token_count: int):
        has_hit_rate_limit = self._rate_limiter.check(total_token_count)
        was_throttled = False
        while not has_hit_rate_limit:
            current_time = time.time()
            time_to_sleep = max(1, 60 - (current_time - self._rate_limiter._last_request_time))
            self.logger.info(f"Rate limit reached. Sleeping for {time_to_sleep} seconds. "
            f"Rate limiter info: {self._rate_limiter}")
            time.sleep(time_to_sleep)
            has_hit_rate_limit = self._rate_limiter.check(total_token_count)
            was_throttled = True
        if was_throttled:
            self.logger.info("Rate limit was hit. So the request was throttled.")
            self._rate_limiter.reset()
            self.logger.info("Rate limit reset now.")

    def run_prompt(self, request: FewShotGptRequest) -> list:
        prompt_message = self.coq_gpt_response_grammar.format_as_per_grammar(request, self._k)
        prompt_message = self.agent_grammar.get_openai_main_message_from_string(prompt_message, "user")
        prompt_messages = [prompt_message]
        prompt_token_count = self._gpt_access.num_tokens_from_messages(prompt_messages)
        messages, total_token_count = self._constrain_tokens_in_history(prompt_message, prompt_token_count, self._max_tokens_per_action)
        success = False
        retries = 10
        time_to_sleep = 60
        exp_factor = 1.25
        tokens_factor = 1.25
        temp_factor = 0.025
        max_temp = 0.4
        temperature = self.temperature
        tokens_to_generate = self._max_tokens_per_action
        upper_bound = 3 * self._max_tokens_per_action
        responses = None
        while not success and retries > 0:
            try:
                self._throttle_if_needed(total_token_count)
                self.logger.info(f"Requesting {total_token_count} tokens.")
                self.logger.info(f"Prompt Message:\n{prompt_message['content']}")
                request_start_time = time.time()
                responses, usage = self._gpt_access.complete_chat(
                    messages,
                    n=self.num_sequences,
                    temperature=temperature,
                    max_tokens=tokens_to_generate,
                    stop=[FewShotGptKeywords.QED])
                request_end_time = time.time()
                time_taken = request_end_time - request_start_time
                apporx_output_tokens = usage["total_tokens"] - total_token_count
                self.logger.debug(f"Request took {time_taken} seconds. Used {usage['total_tokens']} tokens. Approx. output {apporx_output_tokens} tokens.")
                reason = usage["reason"]
                self._rate_limiter.update(usage["total_tokens"], request_start_time, request_end_time)
                success = reason != "length"
                if not success:
                    tokens_to_generate = min(int(tokens_to_generate * tokens_factor), upper_bound)
                    self.logger.info(f"Retrying with {tokens_to_generate} tokens. Earlier response was not complete for reason: {reason}.")
                    self.logger.info(f"Incomplete Response messages: \n{responses}")
                    messages, total_token_count = self._constrain_tokens_in_history(prompt_message, prompt_token_count, tokens_to_generate)
                    # temperature = max(max_temp, temperature + temp_factor)
                    # Don't change the temperature for now
                else:
                    self.logger.debug(f"Got a valid response. Reason: \n{reason}")
                    self.logger.debug(f"Response messages: \n{responses}")
            except InvalidRequestError as e:
                self.logger.info("Got an invalid request error. Not retrying.")
                self.logger.exception(e)
                raise
            except Exception as e:
                self.logger.info("Got an unknown exception. Retrying.")
                self.logger.exception(e)
                time.sleep(time_to_sleep)
                responses = []
                usage = {}
                time_to_sleep *= exp_factor # Exponential backoff
            retries -= 1
        if not success and responses == None:
            # Don't throw an error even with an incomplete response, because the parsing can still make it work.
            raise Exception(f"Failed to get valid response after {retries} tries")
        return responses

    def parse_response(self, responses: list) -> typing.List[typing.Tuple[ProofAction, float]]:
        message_contents =  self.agent_grammar.parse_openai_messages(responses, "assistant")
        actions = []
        total = len(message_contents)
        for idx, message in enumerate(message_contents):
            try:
                coq_gpt_request, parsed_message = self.coq_gpt_request_grammar.get_openai_request(message)
                open_ai_message = self.agent_grammar.get_openai_main_message_from_string(parsed_message, "assistant")
            except Exception as e:
                error = f"Expected {str(e)}"
                error_message = f"Invalid response:\n '{message[0]}', \n Stopping Reason: '{message[1]}'.\n Failure reason: {error} \nPlease respond only in the format specified."
                raise InvalidActionException(error_message)
            probability = (idx + 1) / total # For now just assume that the order of the messages is the order of the actions
            action = coq_gpt_request.action
            action.original_message = open_ai_message
            actions.append((action, probability))
        return actions
    
    def __call__(self, tree_search_action: TreeSearchAction) -> ProofAction:
        pass