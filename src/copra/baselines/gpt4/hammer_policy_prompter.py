#!/usr/bin/env python3

import typing
import os
import logging
from copra.retrieval.coq_bm25_reranker import CoqBM25TrainingDataRetriever
from copra.prompt_generator.agent_grammar import CoqGPTResponseGrammar
from copra.prompt_generator.gpt_request_grammar import CoqGPTRequestGrammar, CoqGptRequestActions
from copra.agent.rate_limiter import RateLimiter
from copra.agent.gpt_guided_tree_search_policy import TreeSearchAction, TreeSearchActionType
from copra.gpts.gpt_access import GptAccess
from copra.gpts.llama_access import LlamaAccess
from itp_interface.rl.proof_action import ProofAction
from copra.prompt_generator.prompter import PolicyPrompter
from copra.prompt_generator.dfs_agent_grammar import DfsAgentGrammar

class HammerPolicyPrompter(PolicyPrompter):
    sledgehammer_command = "show ?thesis sledgehammer"
    coq_hammer_command = "hammer."
    _cache: typing.Dict[str, typing.Any] = {}
    def __init__(self, 
            main_sys_prompt_path: str, 
            example_conv_prompt_path: str,
            k : typing.Optional[int] = None,
            retrieve_prompt_examples: bool = True,
            training_data_path: typing.Optional[str] = None,
            metadata_filename: typing.Optional[str] = None,
            language: ProofAction.Language = ProofAction.Language.ISABELLE,
            logger = None):
        assert language == ProofAction.Language.ISABELLE or language == ProofAction.Language.COQ, f"Language {language} is not supported for hammer policy prompter"
        assert os.path.exists(main_sys_prompt_path), f"{main_sys_prompt_path} doesn't exists"
        assert os.path.exists(example_conv_prompt_path), f"{example_conv_prompt_path} doesn't exists"
        self.agent_grammar = DfsAgentGrammar(user_name="example_user", agent_name="example_assistant")
        self.language = language
        use_defensive_parsing = False
        self.gpt_request_grammar = CoqGPTRequestGrammar(enable_defensive_parsing=use_defensive_parsing)
        self.gpt_response_grammar = CoqGPTResponseGrammar()
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
        pass
   
    def __exit__(self, exc_type, exc_value, traceback):
        pass

    def _init_retriever(self):
        if HammerPolicyPrompter._cache.get(self._training_data_path, None) is not None:
            # Use BM25 from cache if loaded once
            self.logger.info("Using cached BM25 retriever ....")
            self.retriever = HammerPolicyPrompter._cache[self._training_data_path]
        else:
            self.retriever = CoqBM25TrainingDataRetriever(
                self._training_data_path,
                self._metadata_filename,
                k1=1.2,
                b=0.8,
                epsilon=0.1,
                logger=self.logger)
            HammerPolicyPrompter._cache[self._training_data_path] = self.retriever
            self.logger.info("Loading training data for BM25 retriever ....")
            self.retriever.load()
            self.logger.info("Loaded training data for BM25 retriever!")
        self._retrieval_count = 2

    def add_to_history(self, message: typing.Any):
        # Do nothing
        pass

    def run_prompt(self, request) -> list:
        # No matter what the request is, we always return show ?thesis sledgehammer
        self._num_api_calls += 1
        if self.language == ProofAction.Language.COQ:
            message_content = f"[RUN TACTIC]\n{HammerPolicyPrompter.coq_hammer_command}\n"
        elif self.language == ProofAction.Language.ISABELLE:
            message_content = f"[RUN TACTIC]\n{HammerPolicyPrompter.sledgehammer_command}\n"
        else:
            raise Exception(f"Language {self.language} is not supported")
        message = self.agent_grammar.get_openai_main_message_from_string(message_content, "assistant")
        message["finish_reason"] = "stop"
        self.logger.info(f"Command running:\n{message_content}")
        return [message]

    def parse_response(self, responses: list) -> typing.List[typing.Tuple[ProofAction, float]]:
        assert len(responses) == 1, f"Expected only one response, got {len(responses)}"
        message_contents =  self.agent_grammar.parse_openai_messages(responses, "assistant")
        message = message_contents[0]
        gpt_request, parsed_message = self.gpt_request_grammar.get_openai_request(message)
        open_ai_message = self.agent_grammar.get_openai_main_message_from_string(parsed_message, "assistant")
        if gpt_request.action == CoqGptRequestActions.GET_DFNS_THMS:
            action = ProofAction(ProofAction.ActionType.GET_DFNS_THMS, self.language)
        elif gpt_request.action == CoqGptRequestActions.RUN_TACTIC:
            action = ProofAction(ProofAction.ActionType.RUN_TACTIC, self.language, tactics=gpt_request.args)
        else:
            raise Exception(f"Invalid action {gpt_request.action}")
        action.original_message = open_ai_message
        actions = [(action, 1.0)]
        return actions
    
    def __call__(self, tree_search_action: TreeSearchAction) -> ProofAction:
        if tree_search_action.action_type == TreeSearchActionType.FAILED_ACTION_SUMMARY_PROMPT or tree_search_action.action_type == TreeSearchActionType.NEXT_ACTION_SUMMARY_PROMPT:
            # We don't need to do anything here
            responses = self.run_prompt(None) # No need to see the state because the action is always the same
            action_tuple = self.parse_response(responses)
            action = action_tuple[0][0]
            return action
        elif tree_search_action.action_type == TreeSearchActionType.BACKTRACK:
            return ProofAction(ProofAction.ActionType.BACKTRACK, self.language)
        elif tree_search_action.action_type == TreeSearchActionType.STOP:
            return ProofAction(ProofAction.ActionType.EXIT, self.language)
        else:
            raise Exception(f"Invalid action type {tree_search_action.action_type}")

    def get_efficiency_info(self) -> typing.Dict[str, typing.Any]:
        return {
            "api_calls": self._num_api_calls
        }