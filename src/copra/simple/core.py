#!/usr/bin/env python3
"""
Core business logic for simple Lean 4 proof runner.

This module is I/O agnostic and can be used by both CLI and REST API interfaces.
"""

import os
import time
import logging
import typing
from dataclasses import dataclass, field
from typing import Optional, List, Any, Dict
from abc import ABC, abstractmethod

from copra.main.config import (
    EnvSettings, EvalBenchmark, EvalDataset, EvalFile, EvalSettings,
    Experiments, PromptSettings, PolicyName, parse_config
)
from copra.main.policy_factory import PolicyFactory
from copra.main.lemma_discovery import discover_lemmas_with_timeout
from copra.agent.dfs_tree_search_with_stack import DFSTreeSearch
from copra.agent.gpt_guided_tree_search_policy import GptGuidedTreeSearchPolicy
from copra.agent.simple_proof_agent import ProofAgent
from itp_interface.rl.simple_proof_env import ProofEnv
from itp_interface.rl.proof_tree import ProofSearchResult
from itp_interface.rl.proof_action import ProofAction
from itp_interface.tools.proof_exec_callback import ProofExecutorCallback
from itp_interface.tools.misc_defns import HammerMode
from itp_interface.rl.simple_proof_env import ProofEnvReRankStrategy
from itp_interface.tools.log_utils import setup_logger
from omegaconf import OmegaConf


class ProofCallback(ABC):
    """Abstract callback interface for proof execution progress."""

    @abstractmethod
    def on_complete(self, result: ProofSearchResult, execution_time: float) -> None:
        """
        Called when proof execution completes.

        Args:
            result: ProofSearchResult from the proof search
            execution_time: Time taken for proof execution in seconds
        """
        pass

    def on_start(self, theorem_name: str) -> None:
        """Called when proof execution starts."""
        pass


@dataclass
class SimpleLean4Config:
    """Simple configuration for Lean 4 proof runs."""

    # Required parameters
    project: str
    file_path: str
    theorem_name: str

    # Optional parameters with defaults
    timeout: int = 200
    temperature: float = 0.7
    proof_retries: int = 4
    main_prompt: str = "data/prompts/system/simplified-lean4-proof-agent-with-dfs.md"
    conv_prompt: str = "data/prompts/conversation/simplified-lean4-proof-agent-dfs-multiple.md"
    uses_simplified_prompt: bool = True

    # Model configuration
    model_name: str = "gpt-5-mini"
    max_tokens_per_action: int = 4500
    max_steps_per_episode: int = 60

    # Advanced settings (rarely changed)
    policy_name: str = "dfs"
    retrieval_strategy: str = "BM25"
    max_proof_depth: int = 20

    def to_experiments(self) -> Experiments:
        """
        Convert simple config to full Experiments object.

        Returns:
            Experiments object compatible with existing infrastructure
        """
        # Create prompt settings
        prompt_settings = PromptSettings(
            name="simple_lean4",
            main_prompt=self.main_prompt,
            conv_prompt=self.conv_prompt,
            uses_simplified_prompt=self.uses_simplified_prompt,
            informal_proof_repo=None
        )

        # Create eval settings
        eval_settings = EvalSettings(
            name="simple_lean4_eval",
            use_hammer=HammerMode.NONE,
            gpt_model_name=self.model_name,
            temperature=self.temperature,
            timeout_in_secs=self.timeout,
            proof_retries=self.proof_retries,
            max_tokens_per_action=self.max_tokens_per_action,
            max_steps_per_episode=self.max_steps_per_episode,
            max_proof_depth=self.max_proof_depth,
            policy_name=PolicyName.Dfs if self.policy_name.lower() == "dfs" else PolicyName.FewShot,
            should_checkpoint=False,  # No checkpointing in simple mode
            render=False,
            max_number_of_episodes=1,
            sample=1.0,
            sample_seed=0,
            always_use_useful_theorem_retrieval=False,
            proof_dump_dir="",
            model_params={}
        )

        # Create env settings
        env_settings = EnvSettings(
            name="simple_lean4_env",
            retrieval_strategy=ProofEnvReRankStrategy[self.retrieval_strategy.upper()],
        )

        # Create benchmark with single file and theorem
        eval_file = EvalFile(
            path=self.file_path,
            theorems=[self.theorem_name] if self.theorem_name != "*" else "*",
            max_retry_attempts_limits={},
            max_time_limits_in_secs={}
        )

        eval_dataset = EvalDataset(
            project=self.project,
            files=[eval_file]
        )

        eval_benchmark = EvalBenchmark(
            name="simple_lean4",
            language=ProofAction.Language.LEAN4,
            timeout_per_theorem_in_secs=self.timeout,
            num_files=1,
            datasets=[eval_dataset],
            few_shot_data_path_for_retrieval=None,
            few_shot_metadata_filename_for_retrieval=None,
            dfs_data_path_for_retrieval=None,
            dfs_metadata_filename_for_retrieval=None
        )

        return Experiments(
            benchmark=eval_benchmark,
            eval_settings=eval_settings,
            env_settings=env_settings,
            prompt_settings=prompt_settings
        )


class SimpleLean4Runner:
    """Core proof runner - no I/O dependencies."""

    def __init__(self, config: SimpleLean4Config, logger: Optional[logging.Logger] = None):
        """
        Initialize the proof runner.

        Args:
            config: Configuration for the proof run
            logger: Optional logger (creates default if not provided)
        """
        self.config = config
        self.logger = logger or self._create_default_logger()
        self.experiments = config.to_experiments()

    def _create_default_logger(self) -> logging.Logger:
        """Create a default logger if none provided."""
        logger = logging.getLogger("SimpleLean4Runner")
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            ))
            logger.addHandler(handler)
        return logger

    def run_proof(self, callback: Optional[ProofCallback] = None) -> typing.Tuple[ProofSearchResult, float]:
        """
        Execute proof with optional streaming callback.

        Args:
            callback: Optional callback for progress updates

        Returns:
            Tuple of (ProofSearchResult, execution_time_in_seconds)
        """
        start_time = time.time()

        # Get the file and theorem to prove
        dataset = self.experiments.benchmark.datasets[0]
        file = dataset.files[0]
        project_path = dataset.project

        try:
            # Discover lemmas if theorem_name is "*"
            if self.config.theorem_name == "*":
                self.logger.info(f"Discovering lemmas in {self.config.file_path}")
                proof_exec_callback = ProofExecutorCallback(
                    project_path,
                    self.config.file_path,
                    self.experiments.benchmark.language,
                )

                all_lemmas = discover_lemmas_with_timeout(
                    self.config.file_path,
                    proof_exec_callback,
                    timeout=60,
                    logger=self.logger
                )

                if not all_lemmas:
                    error_msg = f"No lemmas discovered in {self.config.file_path}"
                    self.logger.error(error_msg)
                    # Create empty result
                    empty_result = ProofSearchResult(
                        proof_file=None,
                        proof_found=False,
                        lemma_name="*",
                        proof_steps=[],
                        proof_time_in_secs=time.time() - start_time,
                        inferences_taken=0,
                        possible_failed_paths=0,
                        num_of_backtracks=0,
                        is_timeout=False,
                        is_inference_exhausted=False,
                        longest_success_path=0,
                        additional_info={"error": error_msg},
                        language=self.experiments.benchmark.language
                    )
                    execution_time = time.time() - start_time
                    if callback:
                        callback.on_complete(empty_result, execution_time)
                    return empty_result, execution_time

                self.logger.info(f"Discovered {len(all_lemmas)} lemmas: {all_lemmas}")
                # For now, just prove the first lemma
                # TODO: Support proving multiple lemmas
                theorem_name = all_lemmas[0]
                self.logger.info(f"Proving first lemma: {theorem_name}")
            else:
                theorem_name = self.config.theorem_name

            if callback:
                callback.on_start(theorem_name)

            # Run the proof
            result, execution_time = self._run_single_proof(theorem_name, project_path)

            if callback:
                callback.on_complete(result, execution_time)

            return result, execution_time

        except Exception as e:
            self.logger.exception(f"Error during proof execution: {e}")
            # Create error result
            error_result = ProofSearchResult(
                proof_file=None,
                proof_found=False,
                lemma_name=self.config.theorem_name,
                proof_steps=[],
                proof_time_in_secs=time.time() - start_time,
                inferences_taken=0,
                possible_failed_paths=0,
                num_of_backtracks=0,
                is_timeout=False,
                is_inference_exhausted=False,
                longest_success_path=0,
                additional_info={"error": str(e)},
                language=self.experiments.benchmark.language
            )
            execution_time = time.time() - start_time
            if callback:
                callback.on_complete(error_result, execution_time)
            return error_result, execution_time

    def _run_single_proof(
        self,
        theorem_name: str,
        project_path: str
    ) -> typing.Tuple[ProofSearchResult, float]:
        """
        Run proof for a single theorem.

        Args:
            theorem_name: Name of the theorem to prove
            project_path: Path to the project

        Returns:
            Tuple of (ProofSearchResult, execution_time_in_seconds)
        """
        start_time = time.time()

        self.logger.info(f"Starting proof for theorem: {theorem_name}")
        self.logger.info(f"Project: {project_path}")
        self.logger.info(f"File: {self.config.file_path}")

        # Create proof executor callback
        proof_exec_callback = ProofExecutorCallback(
            project_path,
            self.config.file_path,
            self.experiments.benchmark.language,
        )

        # Create policy prompter
        policy_prompter, _, _ = PolicyFactory.create_policy_prompter(
            self.experiments.eval_settings,
            self.experiments.prompt_settings,
            self.experiments.benchmark,
            theorem_name,
            self.logger
        )

        # Create tree search and policy
        dfs_tree_search = DFSTreeSearch(language=self.experiments.benchmark.language)
        search_guidance_policy = GptGuidedTreeSearchPolicy(
            "",  # No checkpoint dir in simple mode
            theorem_name,
            policy_prompter,
            dfs_tree_search,
            checkpoint_on_exit=False,
            language=self.experiments.benchmark.language
        )

        # Create proof environment and agent
        with ProofEnv(
            f"simple_proof_env_{theorem_name}",
            proof_exec_callback,
            theorem_name,
            retrieval_strategy=self.experiments.env_settings.retrieval_strategy,
            max_proof_depth=self.experiments.eval_settings.max_proof_depth,
            always_retrieve_thms=self.experiments.eval_settings.always_use_useful_theorem_retrieval,
            logger=self.logger
        ) as env:
            with search_guidance_policy:
                agent = ProofAgent(
                    f"simple_proof_agent_{theorem_name}",
                    search_guidance_policy,
                    should_checkpoint=False,
                    logger=self.logger
                )

                # Run proof with query limit
                agent.run_episodes_till_stop(
                    env,
                    episodes=self.experiments.eval_settings.max_number_of_episodes,
                    render=False,
                    stop_policy=self._create_stop_policy(),
                    policy_info_message=self._create_info_message()
                )

            # Get proof result
            proof_search_res: ProofSearchResult = env.collect_proof_search_result()
            execution_time = time.time() - start_time

            self.logger.info(f"Proof completed in {execution_time:.2f}s")
            self.logger.info(f"Proof result: {proof_search_res}")

            return proof_search_res, execution_time

    def _create_stop_policy(self) -> typing.Callable[[int, Dict[str, Any]], bool]:
        """Create stop policy based on query limit."""
        max_queries = self.experiments.eval_settings.max_steps_per_episode

        def stop_policy(steps: int, info: Dict[str, Any]) -> bool:
            return info.get("queries", 0) >= max_queries

        return stop_policy

    def _create_info_message(self) -> typing.Callable[[int, Dict[str, Any]], str]:
        """Create info message function for logging."""
        max_queries = self.experiments.eval_settings.max_steps_per_episode

        def info_message(steps: int, info: Dict[str, Any]) -> str:
            queries = info.get("queries", 0)
            return f"Step {queries}/{max_queries} (Actual steps: {steps})"

        return info_message
