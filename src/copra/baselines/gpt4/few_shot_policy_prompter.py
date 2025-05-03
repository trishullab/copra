#!/usr/bin/env python3

import typing
import os
import time
import logging
from copra.retrieval.coq_bm25_reranker import CoqBM25TrainingDataRetriever
from copra.agent.rate_limiter import RateLimiter, InvalidActionException
from copra.agent.gpt_guided_tree_search_policy import TreeSearchAction
from copra.gpts.gpt_access import GptAccess
from copra.gpts.llama_access import LlamaAccess
from itp_interface.rl.proof_action import ProofAction
from copra.prompt_generator.prompter import PolicyPrompter
from copra.prompt_generator.dfs_agent_grammar import DfsAgentGrammar
from copra.baselines.gpt4.few_shot_grammar import FewShotGptRequest, FewShotGptRequestGrammar, FewShotGptResponse, FewShotGptResponseGrammar
from copra.tools.misc import model_supports_openai_api

class FewShotGptPolicyPrompter(PolicyPrompter):
    _cache: typing.Dict[str, typing.Any] = {}
    def __init__(self, 
            main_sys_prompt_path: str, 
            example_conv_prompt_path: str,
            num_sequences: int = 1,
            temperature: float = 0.25,
            max_tokens_per_action: int = 250,
            max_history_messages: int = 0, # This means keep no history of messages
            gpt_model_name: str = "gpt-3.5-turbo",
            secret_filepath: str = None,
            k : typing.Optional[int] = None,
            retrieve_prompt_examples: bool = True,
            training_data_path: typing.Optional[str] = None,
            metadata_filename: typing.Optional[str] = None,
            language: ProofAction.Language = ProofAction.Language.COQ,
            logger = None):
        assert os.path.exists(main_sys_prompt_path), f"{main_sys_prompt_path} doesn't exists"
        assert os.path.exists(example_conv_prompt_path), f"{example_conv_prompt_path} doesn't exists"
        self.agent_grammar = DfsAgentGrammar(user_name="example_user", agent_name="example_assistant")
        self.language = language
        self.model_name = gpt_model_name
        use_defensive_parsing = not gpt_model_name.startswith("gpt")
        self.gpt_request_grammar = FewShotGptRequestGrammar(language, use_defensive_parsing)
        self.gpt_response_grammar = FewShotGptResponseGrammar(language)
        conv_messages = self.agent_grammar.get_openai_conv_messages(example_conv_prompt_path, "system")
        main_message = self.agent_grammar.get_openai_main_message(main_sys_prompt_path, "system")
        self.system_messages = [main_message] + conv_messages
        if not model_supports_openai_api(gpt_model_name):
            self._gpt_access = LlamaAccess(gpt_model_name)
        else:
            self._gpt_access = GptAccess(secret_filepath=secret_filepath, model_name=gpt_model_name)
        self._token_limit_per_min = GptAccess.gpt_model_info[gpt_model_name]["token_limit_per_min"]
        self._request_limit_per_min = GptAccess.gpt_model_info[gpt_model_name]["request_limit_per_min"]
        self._max_token_per_prompt = GptAccess.gpt_model_info[gpt_model_name]["max_token_per_prompt"]
        self._rate_limiter = RateLimiter(self._token_limit_per_min, self._request_limit_per_min)
        self.temperature = temperature
        self.num_sequences = num_sequences
        self.system_token_count = self._gpt_access.num_tokens_from_messages(self.system_messages)
        self._max_tokens_per_action = max_tokens_per_action
        max_token_limit = self._max_token_per_prompt - self._max_tokens_per_action
        assert max_token_limit > 0, f"Max token per prompt {self._max_token_per_prompt} is less than max token per action {self._max_tokens_per_action}"
        assert self.system_token_count < max_token_limit, f"System token count {self.system_token_count} is greater or equal to the max token per prompt {max_token_limit}"
        self._history_token_count = 0
        self._message_history = []
        self._custom_system_messages = []
        self._message_history_token_count = []
        self._max_history_messages = max_history_messages
        self._k = k
        self._retrieve_prompt_examples = retrieve_prompt_examples
        self.logger = logger if logger is not None else logging.getLogger(__name__)
        self._num_api_calls = 0
        self._training_data_path = training_data_path
        self._metadata_filename = metadata_filename
        self.last_message_has_error = False
        if self.language == ProofAction.Language.LEAN or self.language == ProofAction.Language.ISABELLE:
            self._retrieve_prompt_examples = False
        if self._retrieve_prompt_examples:
            assert self._metadata_filename is not None, "Metadata filename must be provided if retrieve_prompt_examples is True"
            assert self._training_data_path is not None, "Training data path must be provided if retrieve_prompt_examples is True"
            assert os.path.exists(self._training_data_path), f"Training data path {self._training_data_path} doesn't exists"
            self._init_retriever()
        pass

    def __enter__(self):
        if isinstance(self._gpt_access, LlamaAccess):
            self._gpt_access.__enter__()
    
    def __exit__(self, exc_type, exc_value, traceback):
        if isinstance(self._gpt_access, LlamaAccess):
            self._gpt_access.__exit__(exc_type, exc_value, traceback)

    def _init_retriever(self):
        if FewShotGptPolicyPrompter._cache.get(self._training_data_path, None) is not None:
            # Use BM25 from cache if loaded once
            self.logger.info("Using cached BM25 retriever ....")
            self.retriever = FewShotGptPolicyPrompter._cache[self._training_data_path]
        else:
            self.retriever = CoqBM25TrainingDataRetriever(
                self._training_data_path,
                self._metadata_filename,
                k1=1.2,
                b=0.8,
                epsilon=0.1,
                logger=self.logger)
            FewShotGptPolicyPrompter._cache[self._training_data_path] = self.retriever
            self.logger.info("Loading training data for BM25 retriever ....")
            self.retriever.load()
            self.logger.info("Loaded training data for BM25 retriever!")
        self._retrieval_count = 2

    def add_to_history(self, message: typing.Any):
        message_token_count = self._gpt_access.num_tokens_from_messages([message])
        self._message_history.append(message)
        self._message_history_token_count.append(message_token_count)
        self._history_token_count += message_token_count

    def _get_prompt_message(self, request: FewShotGptResponse, max_tokens_in_prompt: int):
        assert max_tokens_in_prompt > 0, "Max token per prompt must be greater than 0, please decrease max_tokens_per_action"
        characters_per_token = 5
        prompt_message_tokens_underlimit = False
        if self._retrieve_prompt_examples:
            dfs_with_score = self.retriever.find_relevant_training_data(request.theorem, num_results=self._retrieval_count)
            max_token_for_examples = int(0.25 * max_tokens_in_prompt)
            max_token_for_problem = max_tokens_in_prompt - max_token_for_examples
            retrieved_examples = [dfs for _, dfs in dfs_with_score]
        else:
            max_token_for_examples = 0
            max_token_for_problem = max_tokens_in_prompt
            retrieved_examples = [] 
        # Also return the custom system messages here
        custom_system_messages = []
        custom_system_message_count = 0
        if self._retrieve_prompt_examples:
            custom_message_tokens_underlimit = False
            characters_per_token = 5
            while not custom_message_tokens_underlimit and characters_per_token > 0:
                example_theorems = [
                    retrieved_example.start_goals[0].goal
                    for retrieved_example in retrieved_examples
                ]
                example_theorems = [
                    self.gpt_response_grammar.format_as_per_grammar(
                        FewShotGptResponse(theorem=theorem), 
                        self._k, 
                        max_token_for_examples, 
                        characters_per_token)
                    for theorem in example_theorems
                ]
                example_proofs = [
                    retrieved_example.proof_steps[0]
                    for retrieved_example in retrieved_examples
                ]
                custom_system_messages = []
                char_cnt_remaining = max_token_for_examples * characters_per_token
                for example_theorem, example_proof in zip(example_theorems, example_proofs):
                    char_cnt_theorem = len(example_theorem) + len(example_proof)
                    if  char_cnt_theorem > char_cnt_remaining:
                        # Trim the examples if they are too long
                        example_theorem = example_theorem[:char_cnt_remaining]
                        char_cnt_remaining = char_cnt_remaining - len(example_theorem)
                        example_proof = example_proof[:char_cnt_remaining]
                        char_cnt_remaining = char_cnt_remaining - len(example_proof)
                    else:
                        char_cnt_remaining = char_cnt_remaining - char_cnt_theorem
                    
                    if len(example_theorem) > 0 and len(example_proof) > 0:
                        custom_system_messages.append(
                            self.agent_grammar.get_openai_main_message_from_string(example_theorem, "system", "example_user")
                        )
                        custom_system_messages.append(
                            self.agent_grammar.get_openai_main_message_from_string(example_proof, "system", "example_assistant")
                        )
                    else:
                        break
                custom_system_message_count = self._gpt_access.num_tokens_from_messages(custom_system_messages)
                custom_message_tokens_underlimit = custom_system_message_count < max_token_for_examples
                characters_per_token -= 1
        if custom_system_message_count > max_token_for_examples:
            # Drop the custom system messages if the prompt message is too long
            custom_system_messages = []
            custom_system_message_count = 0
            max_token_for_problem = max_tokens_in_prompt
        characters_per_token = 5
        while not prompt_message_tokens_underlimit and characters_per_token > 0:
            prompt_message = self.gpt_response_grammar.format_as_per_grammar(request, self._k, max_token_for_problem, characters_per_token)
            prompt_message = self.agent_grammar.get_openai_main_message_from_string(prompt_message, "user")
            prompt_messages = [prompt_message]
            prompt_token_count = self._gpt_access.num_tokens_from_messages(prompt_messages)
            prompt_message_tokens_underlimit = prompt_token_count < max_token_for_problem
            characters_per_token -= 1
        assert prompt_token_count <= max_token_for_problem, f"Prompt token count {prompt_token_count} is greater than max token per prompt {max_token_for_problem}"
        assert prompt_token_count + custom_system_message_count <= max_tokens_in_prompt, f"Prompt token count {prompt_token_count} + custom system message token count {custom_system_message_count} is greater than max token per prompt {max_tokens_in_prompt}"
        return prompt_message, prompt_token_count, custom_system_messages, custom_system_message_count

    def _constrain_tokens_in_history(self, prompt_message, custom_example_system_messages : typing.List[dict[str, str]], custom_system_message_count: int, prompt_token_count: int, max_tokens_per_action: int) -> list:
        if len(self._message_history) >= self._max_history_messages:
            history_idx = len(self._message_history) - self._max_history_messages
        else:
            history_idx = 0
        if history_idx < len(self._message_history):
            # There is no point in checking the token count if there is no history to be maintained
            total_token_count = self.system_token_count + self._history_token_count + prompt_token_count + custom_system_message_count
            max_token_per_prompt = min(self._max_token_per_prompt, self._max_token_per_prompt - max_tokens_per_action)
            assert max_token_per_prompt > 0, "Max token per prompt must be greater than 0, please decrease max_tokens_per_action"
            tokens_shredded = False
            remove_cnt  = 0
            history_count = self._history_token_count
            while total_token_count > max_token_per_prompt and history_idx < len(self._message_history):
                self.logger.warning(f"Tokens exceeded removing history at index {history_idx}: {total_token_count} >= {max_token_per_prompt}")
                history_count -= self._message_history_token_count[history_idx]
                total_token_count = self.system_token_count + history_count + prompt_token_count + custom_system_message_count
                history_idx += 1
                tokens_shredded = True
                remove_cnt += 1
            if remove_cnt % 2 == 1 and history_idx < len(self._message_history):
                history_count -= self._message_history_token_count[history_idx]
                total_token_count = self.system_token_count + history_count + prompt_token_count + custom_system_message_count
                history_idx += 1
            if tokens_shredded:
                self.logger.warning(f"Shredded tokens from history. New total token count: {total_token_count}, max token per prompt: {max_token_per_prompt}, history token count: {self._history_token_count}, prompt token count: {prompt_token_count}")
            if total_token_count >= max_token_per_prompt:
                self.logger.warning(f"Total token count {total_token_count} is still greater than max token per prompt {max_token_per_prompt}.")
        else:
            total_token_count = self.system_token_count + prompt_token_count + custom_system_message_count
        if history_idx > 0:
            for idx in range(min(history_idx, len(self._message_history))):
                self._history_token_count -= self._message_history_token_count[idx]
        self._message_history = self._message_history[history_idx:]
        self._message_history_token_count = self._message_history_token_count[history_idx:]
        self._custom_system_messages = custom_example_system_messages
        self._message_history.append(prompt_message)
        self._message_history_token_count.append(prompt_token_count)
        self._history_token_count += prompt_token_count + custom_system_message_count
        messages = self.system_messages + self._custom_system_messages + self._message_history
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
        max_tokens_in_prompt = self._max_token_per_prompt - self.system_token_count - self._max_tokens_per_action
        prompt_message, prompt_token_count, custom_system_msg, custom_system_msg_cnt = self._get_prompt_message(request, max_tokens_in_prompt)
        messages, total_token_count = self._constrain_tokens_in_history(prompt_message, custom_system_msg, custom_system_msg_cnt, prompt_token_count, self._max_tokens_per_action)
        success = False
        retries = 3
        time_to_sleep = 60
        exp_factor = 1.25
        tokens_factor = 1.25
        temp_factor = 0.025
        max_temp = 0.4
        temperature = self.temperature
        tokens_to_generate = self._max_tokens_per_action
        upper_bound = 5 * self._max_tokens_per_action
        responses = None
        while not success and retries > 0:
            try:
                self._throttle_if_needed(total_token_count)
                self.logger.info(f"Requesting {tokens_to_generate} tokens to generate, {total_token_count} tokens in input.")
                if len(custom_system_msg) > 0:
                    self.logger.info(f"Example prompt messages:")
                    for idx, msg in enumerate(custom_system_msg):
                        self.logger.info(f"Example {idx + 1} [{msg['role'], msg['name']}] :\n{msg['content']}")
                self.logger.info(f"Prompt Message:\n{prompt_message['content']}")
                request_start_time = time.time()
                responses, usage = self._gpt_access.complete_chat(
                    messages,
                    n=self.num_sequences,
                    temperature=temperature,
                    max_tokens=tokens_to_generate,
                    stop=[self.gpt_request_grammar.QED])
                request_end_time = time.time()
                time_taken = request_end_time - request_start_time
                apporx_output_tokens = usage["total_tokens"] - total_token_count
                self.logger.debug(f"Request took {time_taken} seconds. Used {usage['total_tokens']} tokens. Approx. output {apporx_output_tokens} tokens.")
                reason = usage["reason"]
                self._rate_limiter.update(usage["total_tokens"], request_start_time, request_end_time)
                success = reason != "length" or tokens_to_generate >= upper_bound
                if not success:
                    tokens_to_generate = min(int(tokens_to_generate * tokens_factor), upper_bound)
                    self.logger.info(f"Retrying with {tokens_to_generate} tokens. Earlier response was not complete for reason: {reason}.")
                    self.logger.info(f"Incomplete Response messages: \n{responses}")
                    max_token_per_prompt = self._max_token_per_prompt - self.system_token_count - tokens_to_generate
                    prompt_message, prompt_token_count, custom_system_msg, custom_system_msg_cnt = self._get_prompt_message(request, max_token_per_prompt) # Re-generate the prompt message within new token limit
                    messages, total_token_count = self._constrain_tokens_in_history(prompt_message, custom_system_msg, custom_system_msg_cnt, prompt_token_count, tokens_to_generate)
                    # temperature = max(max_temp, temperature + temp_factor)
                    # Don't change the temperature for now
                else:
                    if tokens_to_generate >= upper_bound:
                        self.logger.warning(f"Retried {retries} times but still got an incomplete response. Reason: {reason}.")
                        self.logger.info(f"Maxed out response: \n{responses}")
                    else:
                        self.logger.debug(f"Got a valid response. Reason: \n{reason}")
                        self.logger.debug(f"Response messages: \n{responses}")
                self._num_api_calls += 1
            except Exception as e:
                self.logger.info("Got an unknown exception. Retrying.")
                self.logger.exception(e)
                # if not self.model_name.startswith("gpt"):
                #     self.logger.warning("Killing the Llama model.")
                #     LlamaAccess.class_kill()
                #     self.logger.warning("Killed the Llama model.")
                #     self.logger.warning("Restarting the Llama model.")
                #     LlamaAccess.class_init(self.model_name, self.temperature)
                #     self.logger.warning("Restarted the Llama model.")
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
                gpt_request, parsed_message = self.gpt_request_grammar.get_openai_request(message)
                open_ai_message = self.agent_grammar.get_openai_main_message_from_string(parsed_message, "assistant")
            except Exception as e:
                error = f"Expected {str(e)}"
                error_message = f"Invalid response:\n '{message[0]}', \n Stopping Reason: '{message[1]}'.\n Failure reason: {error} \nPlease respond only in the format specified."
                raise InvalidActionException(error_message)
            probability = (idx + 1) / total # For now just assume that the order of the messages is the order of the actions
            action = gpt_request.action
            action.original_message = open_ai_message
            actions.append((action, probability))
        return actions
    
    def __call__(self, tree_search_action: TreeSearchAction) -> ProofAction:
        pass

    def get_efficiency_info(self) -> typing.Dict[str, typing.Any]:
        return {
            "api_calls": self._num_api_calls
        }