#!/usr/bin/env python3

"""
Policy Factory Module

This module provides a factory class for creating proof search policies.
It encapsulates all policy creation logic to avoid code duplication and
improve maintainability.
"""

import os
import logging
from typing import Dict, Any, Optional, Tuple

from copra.baselines.gpt4.hammer_policy_prompter import HammerPolicyPrompter
from copra.agent.dfs_policy_prompter import DfsCoqGptPolicyPrompter
from copra.agent.dfs_tree_search_with_stack import DFSTreeSearch
from copra.agent.dfs_hammer_policy_prompter import HammerDfsIsabelleGptPolicyPrompter
from copra.agent.gpt_guided_tree_search_policy import GptGuidedTreeSearchPolicy
from copra.baselines.gpt4.few_shot_policy import FewShotGptPolicy
from copra.baselines.gpt4.few_shot_policy_prompter import FewShotGptPolicyPrompter
from copra.baselines.gpt4.informal_few_shot_policy import InformalFewShotGptPolicy
from copra.baselines.gpt4.informal_few_shot_policy_prompter import InformalFewShotGptPolicyPrompter
from copra.main.config import EvalBenchmark, EvalSettings, PromptSettings, PolicyName
from copra.prompt_generator.prompter import PolicyPrompter
from itp_interface.rl.abstraction import Policy
from itp_interface.rl.proof_action import ProofAction
from itp_interface.tools.misc_defns import HammerMode


class PolicyFactory:
    """Factory class for creating proof search policies and their configurations."""

    @staticmethod
    def create_policy_prompter(
        eval_settings: EvalSettings,
        prompt_settings: PromptSettings,
        eval_benchmark: EvalBenchmark,
        lemma_name: str,
        logger: logging.Logger
    ) -> Tuple[PolicyPrompter, Optional[Any], Optional[str]]:
        """
        Create a policy prompter based on the evaluation settings.

        Args:
            eval_settings: Evaluation settings containing policy configuration
            prompt_settings: Prompt settings containing prompt paths
            eval_benchmark: Benchmark configuration
            lemma_name: Name of the lemma being proved
            logger: Logger instance

        Returns:
            Tuple of (policy_prompter, informal_proof_repo, informal_proof_dump_directory)
        """
        informal_proof_repo = None
        informal_proof_dump_directory = None

        if eval_settings.policy_name == PolicyName.Dfs:
            policy_prompter = PolicyFactory._create_dfs_policy_prompter(
                eval_settings, prompt_settings, eval_benchmark, lemma_name, logger
            )
            if prompt_settings.informal_proof_repo is not None:
                informal_proof_repo = prompt_settings.get_informal_proof_repo()

        elif eval_settings.policy_name == PolicyName.Hammer:
            policy_prompter = PolicyFactory._create_hammer_policy_prompter(
                eval_settings, prompt_settings, eval_benchmark, logger
            )
            if prompt_settings.informal_proof_repo is not None:
                informal_proof_repo = prompt_settings.get_informal_proof_repo()

        elif eval_settings.policy_name == PolicyName.FewShot:
            policy_prompter = PolicyFactory._create_few_shot_policy_prompter(
                eval_settings, prompt_settings, eval_benchmark, logger
            )
            if prompt_settings.informal_proof_repo is not None:
                informal_proof_repo = prompt_settings.get_informal_proof_repo()

        elif eval_settings.policy_name == PolicyName.InformalFewShot:
            policy_prompter = PolicyFactory._create_informal_few_shot_policy_prompter(
                eval_settings, prompt_settings, eval_benchmark, logger
            )
            informal_proof_repo = prompt_settings.get_informal_proof_repo()
            informal_proof_dump_directory = os.path.join(
                eval_settings.proof_dump_dir, "informal_proofs"
            )
            os.makedirs(informal_proof_dump_directory, exist_ok=True)

        else:
            raise Exception(f"Unknown policy name: {eval_settings.policy_name}")

        return policy_prompter, informal_proof_repo, informal_proof_dump_directory

    @staticmethod
    def create_policy_config(
        eval_settings: EvalSettings,
        eval_benchmark: EvalBenchmark,
        prompt_settings: PromptSettings,
        lemma_name: str
    ) -> Dict[str, Any]:
        """
        Create a serializable policy configuration dictionary for multiprocessing.

        This is designed to be picklable for Python 3.14t's forkserver multiprocessing.
        Instead of passing the policy_prompter object (which contains unpicklable thread locks),
        we pass the configuration needed to recreate it inside the subprocess.

        Args:
            eval_settings: Evaluation settings
            eval_benchmark: Benchmark configuration
            prompt_settings: Prompt settings
            lemma_name: Name of the lemma being proved

        Returns:
            Dictionary containing policy configuration
        """
        config = {
            'policy_name': eval_settings.policy_name,
            'eval_benchmark_language': eval_benchmark.language,
            'checkpoint_dir': eval_settings.checkpoint_dir,
            'should_checkpoint': eval_settings.should_checkpoint,
            # Configuration to recreate policy_prompter and related objects inside subprocess
            'eval_settings': eval_settings,
            'prompt_settings': prompt_settings,
            'eval_benchmark': eval_benchmark,
            'lemma_name': lemma_name,
        }

        return config

    @staticmethod
    def _create_dfs_policy_prompter(
        eval_settings: EvalSettings,
        prompt_settings: PromptSettings,
        eval_benchmark: EvalBenchmark,
        lemma_name: str,
        logger: logging.Logger
    ) -> PolicyPrompter:
        """Create DFS policy prompter."""
        informal_proof_repo = None
        if prompt_settings.informal_proof_repo is not None:
            informal_proof_repo = prompt_settings.get_informal_proof_repo()

        # Determine prompter class
        if eval_settings.use_hammer == HammerMode.ALWAYS and \
           eval_benchmark.language == ProofAction.Language.ISABELLE:
            policy_prompter_class = HammerDfsIsabelleGptPolicyPrompter
        else:
            policy_prompter_class = DfsCoqGptPolicyPrompter

        # Common arguments
        common_args = {
            'main_sys_prompt_path': prompt_settings.main_prompt,
            'example_conv_prompt_path': prompt_settings.conv_prompt,
            'max_tokens_per_action': eval_settings.max_tokens_per_action,
            'gpt_model_name': eval_settings.gpt_model_name,
            'temperature': eval_settings.temperature,
            'max_history_messages': eval_settings.max_history_messages,
            'k': eval_settings.max_theorems_in_prompt,
            'retrieve_prompt_examples': eval_settings.use_example_retrieval,
            'num_goal_per_prompt': eval_settings.num_goal_per_prompt,
            'training_data_path': eval_benchmark.dfs_data_path_for_retrieval,
            'metadata_filename': eval_benchmark.dfs_metadata_filename_for_retrieval,
            'language': eval_benchmark.language,
            'logger': logger,
            'informal_proof_repo': informal_proof_repo,
            'lemma_name': lemma_name,
        }

        # Add model_params if provided
        if len(eval_settings.model_params) > 0:
            common_args['model_params'] = eval_settings.model_params

        return policy_prompter_class(**common_args)

    @staticmethod
    def _create_hammer_policy_prompter(
        eval_settings: EvalSettings,
        prompt_settings: PromptSettings,
        eval_benchmark: EvalBenchmark,
        logger: logging.Logger
    ) -> PolicyPrompter:
        """Create Hammer policy prompter."""
        return HammerPolicyPrompter(
            main_sys_prompt_path=prompt_settings.main_prompt,
            example_conv_prompt_path=prompt_settings.conv_prompt,
            k=eval_settings.max_theorems_in_prompt,
            retrieve_prompt_examples=eval_settings.use_example_retrieval,
            training_data_path=eval_benchmark.dfs_data_path_for_retrieval,
            metadata_filename=eval_benchmark.dfs_metadata_filename_for_retrieval,
            language=eval_benchmark.language,
            logger=logger
        )

    @staticmethod
    def _create_few_shot_policy_prompter(
        eval_settings: EvalSettings,
        prompt_settings: PromptSettings,
        eval_benchmark: EvalBenchmark,
        logger: logging.Logger
    ) -> PolicyPrompter:
        """Create FewShot policy prompter."""
        return FewShotGptPolicyPrompter(
            main_sys_prompt_path=prompt_settings.main_prompt,
            example_conv_prompt_path=prompt_settings.conv_prompt,
            temperature=eval_settings.temperature,
            max_tokens_per_action=eval_settings.max_tokens_per_action,
            max_history_messages=eval_settings.max_history_messages,
            gpt_model_name=eval_settings.gpt_model_name,
            k=eval_settings.max_theorems_in_prompt,
            retrieve_prompt_examples=eval_settings.use_example_retrieval,
            training_data_path=eval_benchmark.few_shot_data_path_for_retrieval,
            metadata_filename=eval_benchmark.few_shot_metadata_filename_for_retrieval,
            language=eval_benchmark.language,
            logger=logger
        )

    @staticmethod
    def _create_informal_few_shot_policy_prompter(
        eval_settings: EvalSettings,
        prompt_settings: PromptSettings,
        eval_benchmark: EvalBenchmark,
        logger: logging.Logger
    ) -> PolicyPrompter:
        """Create InformalFewShot policy prompter."""
        return InformalFewShotGptPolicyPrompter(
            main_sys_prompt_path=prompt_settings.main_prompt,
            example_conv_prompt_path=prompt_settings.conv_prompt,
            temperature=eval_settings.temperature,
            max_tokens_per_action=eval_settings.max_tokens_per_action,
            max_history_messages=eval_settings.max_history_messages,
            gpt_model_name=eval_settings.gpt_model_name,
            k=eval_settings.max_theorems_in_prompt,
            retrieve_prompt_examples=eval_settings.use_example_retrieval,
            training_data_path=eval_benchmark.few_shot_data_path_for_retrieval,
            metadata_filename=eval_benchmark.few_shot_metadata_filename_for_retrieval,
            language=eval_benchmark.language,
            logger=logger
        )
