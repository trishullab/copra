#!/usr/bin/env python3

import copy
import typing
import os
import time
import logging
from itp_interface.rl.proof_action import ProofAction
from itp_interface.rl.simple_proof_env import ProgressState
from copra.agent.rate_limiter import RateLimiter, InvalidActionException
from copra.agent.gpt_guided_tree_search_policy import PromptSummary, ProofQInfo, TreeSearchAction, TreeSearchActionType
from copra.gpts.gpt_access import GptAccess
from copra.gpts.llama_access import LlamaAccess, ServiceDownError
from copra.retrieval.coq_bm25_reranker import CoqBM25TrainingDataRetriever
from copra.prompt_generator.prompter import PolicyPrompter
from copra.prompt_generator.gpt_request_grammar import CoqGPTRequestGrammar, CoqGptRequest, CoqGptRequestActions
from copra.prompt_generator.dfs_agent_grammar import DfsAgentGrammar
from copra.prompt_generator.dfs_gpt_response_grammar import CoqGPTResponseDfsGrammar, CoqGptResponse, CoqGptResponseActions
from copra.tools.informal_proof_repo import InformalProofRepo
from copra.tools.misc import model_supports_openai_api

class DfsCoqGptPolicyPrompter(PolicyPrompter):
    _cache: typing.Dict[str, typing.Any] = {}
    def __init__(self, 
            main_sys_prompt_path: str, 
            example_conv_prompt_path: str,
            num_sequences: int = 1,
            temperature: float = 0.25,
            max_tokens_per_action: int = 50,
            max_history_messages: int = 0, # This means keep no history of messages
            gpt_model_name: str = "gpt-3.5-turbo",
            secret_filepath: str = None,
            k : typing.Optional[int] = None,
            retrieve_prompt_examples: bool = True,
            num_goal_per_prompt: typing.Optional[int] = None,
            training_data_path: typing.Optional[str] = None,
            metadata_filename: typing.Optional[str] = None,
            language: ProofAction.Language = ProofAction.Language.COQ,
            logger = None,
            informal_proof_repo: typing.Optional[InformalProofRepo] = None,
            lemma_name: typing.Optional[str] = None,
            model_params: typing.Optional[typing.Dict[str, typing.Any]] = None):
        assert os.path.exists(main_sys_prompt_path), f"{main_sys_prompt_path} doesn't exists"
        assert os.path.exists(example_conv_prompt_path), f"{example_conv_prompt_path} doesn't exists"
        self.agent_grammar = DfsAgentGrammar(user_name="example_user", agent_name="example_assistant")
        self.model_name = gpt_model_name
        use_defensive_parsing = not model_supports_openai_api(gpt_model_name)
        self.coq_gpt_request_grammar = CoqGPTRequestGrammar(enable_defensive_parsing=use_defensive_parsing)
        self.coq_gpt_response_grammar = CoqGPTResponseDfsGrammar()
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
        self._model_params = model_params if model_params is not None else {}
        self._max_tokens_per_action = max_tokens_per_action
        self._history_token_count = 0
        self._message_history = []
        self._message_history_token_count = []
        self._custom_system_messages = []
        self._max_history_messages = max_history_messages
        self._k = k
        self._retrieve_prompt_examples = retrieve_prompt_examples
        self.logger = logger if logger is not None else logging.getLogger(__name__)
        self._num_api_calls = 0
        self._training_data_path = training_data_path
        self._metadata_filename = metadata_filename
        self._num_goal_per_prompt = num_goal_per_prompt
        self.language = language
        self.incorrect_repeat_count = 0 # 1 # Give only one warning
        self.incorrect_repeat_warning = "warning: You are trying to repeat the same incorrect step. Please try a different step, otherwise this will lead to backtracking or termination of proof search. Only repeat if you have run out of all other options, and want to backtrack to the previous state."
        self.last_message_has_error = False
        self.informal_proof_repo = informal_proof_repo
        self.lemma_name = lemma_name
        if self.informal_proof_repo is not None:
            assert self.lemma_name is not None, "Lemma name must be provided if informal proof repo is provided"
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
        if DfsCoqGptPolicyPrompter._cache.get(self._training_data_path, None) is not None:
            # Use BM25 from cache if loaded once
            self.logger.info("Using cached BM25 retriever ....")
            self.retriever = DfsCoqGptPolicyPrompter._cache[self._training_data_path]
        else:
            self.retriever = CoqBM25TrainingDataRetriever(
                self._training_data_path,
                self._metadata_filename,
                k1=1.2,
                b=0.8,
                epsilon=0.1,
                logger=self.logger)
            DfsCoqGptPolicyPrompter._cache[self._training_data_path] = self.retriever
            self.logger.info("Loading training data for BM25 retriever ....")
            self.retriever.load()
            self.logger.info("Loaded training data for BM25 retriever!")
        self._retrieval_count = 2

    def add_to_history(self, message: typing.Any):
        message_token_count = self._gpt_access.num_tokens_from_messages([message])
        self._message_history.append(message)
        self._message_history_token_count.append(message_token_count)
        self._history_token_count += message_token_count
    
    def reset_last_message(self, message: typing.Any):
        if len(self._message_history) > 0:
            self._history_token_count -= self._message_history_token_count[-1]
            self._message_history.pop()
            self._message_history_token_count.pop()
        self.add_to_history(message)

    def _constrain_tokens_in_history(self, prompt_message, custom_example_system_messages : typing.List[dict[str, str]], custom_system_message_count: int, prompt_token_count: int, max_tokens_per_action: int) -> list:
        if len(self._message_history) >= self._max_history_messages:
            if not self.last_message_has_error:
                history_idx = len(self._message_history) - self._max_history_messages
            else:
                history_idx = 0
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
            while total_token_count >= max_token_per_prompt and history_idx < len(self._message_history):
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
        assert total_token_count + max_tokens_per_action <= self._max_token_per_prompt, f"Total token count {total_token_count} + max tokens per action {max_tokens_per_action} is greater than max token per prompt {self._max_token_per_prompt}"
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

    def _get_prompt_message(self, request: CoqGptResponse, max_tokens_in_prompt: int) -> str:
        assert max_tokens_in_prompt > 0, "Max token per prompt must be greater than 0, please decrease max_tokens_per_action"
        retrieve_prompt_examples = self._retrieve_prompt_examples and request.training_data_format is not None and len(request.training_data_format.start_goals) > 0

        if self.informal_proof_repo is not None:
            request.informal_theorem, request.informal_proof = self.informal_proof_repo.get_informal_thm_proof(self.lemma_name)

        if self._num_goal_per_prompt is not None and request.training_data_format is not None and len(request.training_data_format.start_goals) > 0:
            num_goals = max(min(self._num_goal_per_prompt, len(request.training_data_format.start_goals)), 1)
            request = copy.deepcopy(request)
            request.training_data_format.start_goals = request.training_data_format.start_goals[:num_goals]
        if retrieve_prompt_examples:
            dfs_with_score = self.retriever.find_relevant_training_data(
                request.training_data_format.start_goals[0].goal, # Just focus on the first goal for now
                num_results=self._retrieval_count)
            # Sort by score
            dfs_with_score = sorted(dfs_with_score, key=lambda x: x[1], reverse=True)
            max_token_for_examples = int(0.25 * max_tokens_in_prompt)
            max_token_for_problem = max_tokens_in_prompt - max_token_for_examples
            retrieved_examples = [dfs for _, dfs in dfs_with_score]
            # Remove any useful theorems and relevant definitions
            for ex in retrieved_examples:
                for goal in ex.start_goals:
                    goal.possible_useful_theorems_external = []
                    goal.possible_useful_theorems_local = []
                    goal.relevant_defns = []
                ex.all_useful_defns_theorems = []
        else:
            max_token_for_examples = 0
            max_token_for_problem = max_tokens_in_prompt
            retrieved_examples = []
        # Also return the custom system messages here
        custom_system_messages = []
        custom_system_message_count = 0
        characters_per_token = 4.0
        if retrieve_prompt_examples:
            full_example_theorems = [
                self.coq_gpt_response_grammar.format_as_per_grammar(
                    CoqGptResponse(training_data_format=tdf), 
                    self._k, 
                    max_token_cnt=None, 
                    characters_per_token=characters_per_token)
                for tdf in retrieved_examples
            ]
            full_example_proofs = [
                self.coq_gpt_request_grammar.generate_message_from_gpt_request(
                    CoqGptRequest(CoqGptRequestActions.RUN_TACTIC, args=retrieved_example.proof_steps))
                for retrieved_example in retrieved_examples
            ]
            example_char_cnt = sum([len(theorem) + len(proof) for theorem, proof in zip(full_example_theorems, full_example_proofs)])
            full_example_messages = []
            for theorem, proof in zip(full_example_theorems, full_example_proofs):
                theorem_message = self.agent_grammar.get_openai_main_message_from_string(theorem, "system", "example_user")
                proof_message = self.agent_grammar.get_openai_main_message_from_string(proof, "system", "example_assistant")
                full_example_messages.append(theorem_message)
                full_example_messages.append(proof_message)
            example_token_cnt = self._gpt_access.num_tokens_from_messages(full_example_messages)
            characters_per_token = example_char_cnt / example_token_cnt
            assert characters_per_token > 0, f"Characters per token is {characters_per_token}"
            custom_system_messages = full_example_messages
            retries = 50
            decrement_factor = 0.2
            characters_per_token = example_char_cnt / max_token_for_examples
            characters_per_token -= decrement_factor
            assert (characters_per_token < 0 and example_token_cnt > max_token_for_examples) or characters_per_token > 0, f"Characters per token is {characters_per_token} for {example_char_cnt} characters and {example_token_cnt} tokens, and max token for examples is {max_token_for_examples}"
            while example_token_cnt > max_token_for_examples and retries > 0 and characters_per_token > 0:
                example_theorems = [
                    self.coq_gpt_response_grammar.format_as_per_grammar(
                        CoqGptResponse(training_data_format=tdf), 
                        self._k, 
                        max_token_for_examples, 
                        characters_per_token)
                    for tdf in retrieved_examples
                ]
                example_proofs = [
                    self.coq_gpt_request_grammar.generate_message_from_gpt_request(
                        CoqGptRequest(CoqGptRequestActions.RUN_TACTIC, args=retrieved_example.proof_steps))
                    for retrieved_example in retrieved_examples
                ]
                custom_system_messages = []
                token_cnt_till_now = 0
                example_char_cnt = 0
                for example_theorem, example_proof in zip(example_theorems, example_proofs):
                    theorem_message = self.agent_grammar.get_openai_main_message_from_string(example_theorem, "system", "example_user")
                    proof_message = self.agent_grammar.get_openai_main_message_from_string(example_proof, "system", "example_assistant")
                    token_cnt = self._gpt_access.num_tokens_from_messages([theorem_message, proof_message])
                    if token_cnt_till_now + token_cnt >= max_token_for_examples:
                        break
                    else:
                        example_char_cnt += len(example_theorem) + len(example_proof)
                        token_cnt_till_now += token_cnt
                        custom_system_messages.append(theorem_message)
                        custom_system_messages.append(proof_message)
                if example_char_cnt == 0:
                    break # no examples selected
                retries -= 1
                characters_per_token -= decrement_factor
                example_token_cnt = self._gpt_access.num_tokens_from_messages(custom_system_messages)
                if example_token_cnt > max_token_for_examples:
                    self.logger.warning(f"Example token count {example_token_cnt} is greater than max token per example {max_token_for_examples}. Retrying with {characters_per_token} characters per token.")
            custom_system_message_count = self._gpt_access.num_tokens_from_messages(custom_system_messages)
        if custom_system_message_count > max_token_for_examples:
            # Drop the custom system messages if the prompt message is too long
            custom_system_messages = []
            custom_system_message_count = 0
            max_token_for_problem = max_tokens_in_prompt

        characters_per_token = 4.0
        full_prompt_message = self.coq_gpt_response_grammar.format_as_per_grammar(request, self._k, max_token_cnt=None, characters_per_token=characters_per_token)
        prompt_char_cnt = len(full_prompt_message)
        full_prompt_message = self.agent_grammar.get_openai_main_message_from_string(full_prompt_message, "user")
        prompt_token_count = self._gpt_access.num_tokens_from_messages([full_prompt_message])
        characters_per_token = prompt_char_cnt / prompt_token_count
        decrement_factor = 0.1
        characters_per_token -= decrement_factor
        retries = 50
        prompt_message = full_prompt_message
        prompt_messages = [full_prompt_message]
        assert (characters_per_token < 0 and prompt_token_count > max_token_for_problem) or characters_per_token > 0, f"Characters per token is {characters_per_token} for {prompt_char_cnt} characters and {prompt_token_count} tokens, and max token for problem is {max_token_for_problem}"
        while prompt_token_count > max_token_for_problem and retries > 0 and characters_per_token > 0:
            prompt_message = self.coq_gpt_response_grammar.format_as_per_grammar(request, self._k, max_token_for_problem, characters_per_token)
            prompt_char_cnt = len(prompt_message)
            prompt_message = self.agent_grammar.get_openai_main_message_from_string(prompt_message, "user")
            prompt_messages = [prompt_message]
            prompt_token_count = self._gpt_access.num_tokens_from_messages(prompt_messages)
            retries -= 1
            characters_per_token -= decrement_factor
            if prompt_token_count > max_token_for_problem:
                self.logger.warning(f"Prompt token count {prompt_token_count} is greater than max token per prompt {max_token_for_problem}. Retrying with {characters_per_token} characters per token.")
            assert prompt_char_cnt > 0, f"Prompt message is empty. Please decrease max_tokens_per_action. Current value: {self._max_tokens_per_action}"

        prompt_token_count = self._gpt_access.num_tokens_from_messages(prompt_messages)
        assert prompt_token_count <= max_token_for_problem, f"Prompt token count {prompt_token_count} is greater than max token per prompt {max_token_for_problem}"
        assert prompt_token_count + custom_system_message_count <= max_tokens_in_prompt, f"Prompt token count {prompt_token_count} + custom system message token count {custom_system_message_count} is greater than max token per prompt {max_tokens_in_prompt}"
        return prompt_message, prompt_token_count, custom_system_messages, custom_system_message_count

    def run_prompt(self, request: CoqGptResponse) -> list:
        max_tokens_in_prompt = self._max_token_per_prompt - self.system_token_count - self._max_tokens_per_action
        prompt_message, prompt_token_count, custom_system_msg, custom_system_msg_cnt = self._get_prompt_message(request, max_tokens_in_prompt)
        messages, total_token_count = self._constrain_tokens_in_history(prompt_message, custom_system_msg, custom_system_msg_cnt, prompt_token_count, self._max_tokens_per_action)
        success = False
        retries = 6
        time_to_sleep = 60
        exp_factor = 1.06
        tokens_factor = 1.75
        temp_factor = 0.025
        max_temp = 0.4
        temperature = self.temperature
        tokens_to_generate = self._max_tokens_per_action
        upper_bound = 10 * self._max_tokens_per_action
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
                if len(self._model_params) > 0:
                    responses, usage = self._gpt_access.complete_chat(
                        messages,
                        n=self.num_sequences,
                        temperature=temperature,
                        max_tokens=tokens_to_generate,
                        stop=["[END]"],
                        **self._model_params)
                else:
                    responses, usage = self._gpt_access.complete_chat(
                        messages,
                        n=self.num_sequences,
                        temperature=temperature,
                        max_tokens=tokens_to_generate,
                        stop=["[END]"])
                request_end_time = time.time()
                time_taken = request_end_time - request_start_time
                apporx_output_tokens = usage["total_tokens"] - total_token_count
                self.logger.info(f"Request took {time_taken} seconds. Used {usage['total_tokens']} tokens. Used {usage['completion_tokens']} completion tokens. Approx. output {apporx_output_tokens} tokens.")
                reason = usage["reason"]
                self._rate_limiter.update(usage["total_tokens"], request_start_time, request_end_time)
                success = reason != "length" or tokens_to_generate >= upper_bound
                if not success:
                    tokens_to_generate = min(int(tokens_to_generate * tokens_factor), upper_bound)
                    self.logger.info(f"Retrying with {tokens_to_generate} tokens. Earlier response was not complete for reason: {reason}.  Used {usage['completion_tokens']} completion tokens.")
                    self.logger.info(f"Incomplete Response messages: \n{responses}")
                    max_token_per_prompt = self._max_token_per_prompt - self.system_token_count - tokens_to_generate
                    prompt_message, prompt_token_count, custom_system_msg, custom_system_msg_cnt = self._get_prompt_message(request, max_token_per_prompt) # Re-generate the prompt message within new token limit
                    messages, total_token_count = self._constrain_tokens_in_history(prompt_message, custom_system_msg, custom_system_msg_cnt, prompt_token_count, tokens_to_generate)
                    # temperature = max(max_temp, temperature + temp_factor)
                    # don't change temperature for now
                else:
                    if tokens_to_generate >= upper_bound:
                        self.logger.warning(f"Retried {retries} times but still got an incomplete response. Reason: {reason}.")
                        self.logger.info(f"Maxed out response: \n{responses}")
                    else:
                        self.logger.debug(f"Got a valid response. Reason: \n{reason}")
                        self.logger.debug(f"Response messages: \n{responses}")
                self._num_api_calls += 1
            except ServiceDownError as e:
                self.logger.info("Got a service down error. Will giveup until the docker container is restarted.")
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
                error_message = f"Invalid response:\n '{message[0]}'.\n Failure reason: {error} \nPlease respond only in the format specified."
                raise InvalidActionException(error_message)
            probability = (idx + 1) / total # For now just assume that the order of the messages is the order of the actions
            if coq_gpt_request.action == CoqGptRequestActions.GET_DFNS_THMS:
                action = ProofAction(ProofAction.ActionType.GET_DFNS_THMS, self.language)
            elif coq_gpt_request.action == CoqGptRequestActions.RUN_TACTIC:
                action = ProofAction(ProofAction.ActionType.RUN_TACTIC, self.language, tactics=coq_gpt_request.args)
            else:
                raise Exception(f"Invalid action {coq_gpt_request.action}")
            action.original_message = open_ai_message
            actions.append((action, probability))
        return actions
    
    def reset_last_action(self, last_action: ProofAction):
        # Reset the messages in the history
        original_message = last_action.original_message
        if original_message is not None:
            self.reset_last_message(original_message)

    def __call__(self, tree_search_action: TreeSearchAction) -> ProofAction:
        state = tree_search_action.state
        assert state is not None
        assert tree_search_action.kwargs is not None and "summary" in tree_search_action.kwargs
        prompt_summary : PromptSummary = tree_search_action.kwargs["summary"]
        actions_till_now = prompt_summary.actions_till_now
        # Fix the bug here, about none type object
        steps = self.coq_gpt_request_grammar.parse_request_to_args([action.original_message["content"] for action in actions_till_now])
        last_action = prompt_summary.last_action
        last_step = None if last_action is None else self.coq_gpt_request_grammar.parse_request_to_args([last_action.original_message["content"]])[0]
        qinfo: ProofQInfo = prompt_summary.state_info.qinfo
        env_info = qinfo.proof_env_info if qinfo is not None else None
        incorrect_actions = copy.deepcopy(prompt_summary.incorrect_actions)
        # don't show the last action as incorrect becuase it has already been shown as incorrect
        if len(incorrect_actions) > 0 and last_action in incorrect_actions:
            incorrect_actions.remove(last_action)
        incorrect_steps = [action.original_message["content"] for action in incorrect_actions]
        incorrect_steps = self.coq_gpt_request_grammar.parse_request_to_args(incorrect_steps)
        if tree_search_action.action_type == TreeSearchActionType.NEXT_ACTION_SUMMARY_PROMPT:
            # assert len(incorrect_actions) == 0, "There are some incorrect steps. We cannot go to the next action with incorrect steps."
            gpt_response = CoqGptResponse(CoqGptResponseActions.GOALS,
                success=True,
                steps=steps,
                last_step=last_step,
                incorrect_steps=incorrect_steps,
                error_message=None,
                training_data_format=state.training_data_format)
        elif tree_search_action.action_type == TreeSearchActionType.FAILED_ACTION_SUMMARY_PROMPT:
            assert env_info is not None
            assert env_info.progress == ProgressState.FAILED
            assert env_info.error_message is not None
            gpt_response = CoqGptResponse(CoqGptResponseActions.GOALS,
                success = False,
                message = env_info.error_message,
                steps=steps,
                last_step = last_step,
                incorrect_steps=incorrect_steps,
                error_message=env_info.error_message,
                training_data_format=state.training_data_format)
        elif tree_search_action.action_type == TreeSearchActionType.BACKTRACK:
            return ProofAction(ProofAction.ActionType.BACKTRACK, self.language)
        elif tree_search_action.action_type == TreeSearchActionType.STOP:
            return ProofAction(ProofAction.ActionType.EXIT, self.language)
        else:
            raise Exception(f"Invalid action type {tree_search_action.action_type}")
        success = False
        tries = 10
        exceptions = []
        incorrect_action_repeat_count = 0
        while not success and tries > 0:
            try:
                responses = self.run_prompt(gpt_response)
                actions_tuple = self.parse_response(responses)
                chosen_message = actions_tuple[0][0].original_message # Selecting only top action here
                self.add_to_history(chosen_message)
                if self.last_message_has_error:
                    self.last_message_has_error = False

                if (self.incorrect_repeat_count > 0) and (len(incorrect_steps) > 0 or (gpt_response.last_step is not None and not gpt_response.success)):
                    # Create invalid requests first and then match with the chosen one
                    invalid_requests = []
                    invalid_messages = set()                    
                    for incorrect_step in incorrect_steps:
                        if incorrect_step.startswith(CoqGptRequestActions.GET_DFNS_THMS[1:-1]):
                            invalid_requests.append(CoqGptRequest(CoqGptRequestActions.GET_DFNS_THMS, args=[]))
                        else:
                            invalid_requests.append(CoqGptRequest(CoqGptRequestActions.RUN_TACTIC, args=[incorrect_step]))
                    if gpt_response.last_step is not None and not gpt_response.success:
                        if gpt_response.last_step.startswith(CoqGptRequestActions.GET_DFNS_THMS[1:-1]):
                            invalid_requests.append(CoqGptRequest(CoqGptRequestActions.GET_DFNS_THMS, args=[]))
                        else:
                            invalid_requests.append(CoqGptRequest(CoqGptRequestActions.RUN_TACTIC, args=[gpt_response.last_step]))
                    invalid_messages = set([self.coq_gpt_request_grammar.generate_message_from_gpt_request(request) for request in invalid_requests])
                    if chosen_message['content'] in invalid_messages and incorrect_action_repeat_count < self.incorrect_repeat_count:
                        incorrect_action_repeat_count += 1
                        temp_action = actions_tuple[0][0]
                        temp_action_str = temp_action.original_message["content"]
                        if temp_action_str.startswith(CoqGptRequestActions.GET_DFNS_THMS[1:-1]):
                            temp_action_str = CoqGptRequestActions.GET_DFNS_THMS[1:-1]
                        else:
                            temp_action_str = temp_action_str[len(CoqGptRequestActions.RUN_TACTIC):-len(CoqGPTRequestGrammar.end)]
                        incorrect_message = self.incorrect_repeat_warning + f"\nincorrect step:\n {temp_action_str}"
                        self.logger.warning(incorrect_message)
                        # Add the warning message
                        if gpt_response.error_message is not None:
                            gpt_response.error_message = f"{incorrect_message}\n{gpt_response.error_message}"
                        else:
                            gpt_response.error_message = incorrect_message
                    else:
                        incorrect_action_repeat_count = 0
                success = incorrect_action_repeat_count == 0
            except InvalidActionException as e:
                gpt_response = CoqGptResponse(action = CoqGptResponseActions.ERROR, 
                message=e.message)
                chosen_message = responses[0]
                self.last_message_has_error = True
                self.add_to_history(chosen_message)
                exceptions.append(e)
            tries -= 1
        if not success:
            raise Exception(f"Failed to get valid action after {tries} tries. Exceptions:\n {exceptions}")
        action = actions_tuple[0][0]
        return action

    def get_efficiency_info(self) -> typing.Dict[str, typing.Any]:
        return {
            "api_calls": self._num_api_calls
        }