#!/usr/bin/env python3

"""
Proof Execution Module

This module handles the execution of proof attempts with multiprocessing,
timeout management, and retry logic for service failures.
"""

import logging
import time
import typing
from typing import Dict, Any

from copra.main.parallel_execution import get_executor
from copra.gpts.llama_access import ServiceDownError
from copra.agent.dfs_tree_search_with_stack import DFSTreeSearch
from copra.agent.gpt_guided_tree_search_policy import GptGuidedTreeSearchPolicy
from copra.agent.simple_proof_agent import ProofAgent
from copra.baselines.gpt4.few_shot_policy import FewShotGptPolicy
from copra.baselines.gpt4.informal_few_shot_policy import InformalFewShotGptPolicy
from copra.main.config import EnvSettings, EvalSettings, PolicyName
from itp_interface.rl.simple_proof_env import ProofEnv
from itp_interface.rl.proof_tree import ProofSearchResult
from itp_interface.tools.proof_exec_callback import ProofExecutorCallback


def check_query_limit_reached(max_query_limit: int) -> typing.Callable[[int, typing.Dict[str, typing.Any]], bool]:
    """Create a function that checks if query limit has been reached."""
    def _check_query_limit_reached(steps: int, info: typing.Dict[str, typing.Any]):
        return info["queries"] >= max_query_limit
    return _check_query_limit_reached


def query_limit_info_message(max_query_limit: int) -> typing.Callable[[int, typing.Dict[str, typing.Any]], str]:
    """Create a function that formats query limit info message."""
    def _query_limit_info_message(steps: int, info: typing.Dict[str, typing.Any]):
        return f"Step {info['queries']}/{max_query_limit} (Actual steps: {steps})"
    return _query_limit_info_message


# Module-level wrapper for running prover - must be at module level for Python 3.14t pickling
def _run_prover_wrapper(
    ret_dict: Dict[str, Any],
    lemma_name: str,
    proof_exec_callback: ProofExecutorCallback,
    env_settings: EnvSettings,
    eval_settings: EvalSettings,
    policy_config: Dict[str, Any],
    proof_dump_file_name: str,
    log_dir: str,
    theorem_idx: int,
    path: str
) -> None:
    """
    Wrapper function to run prover - must be at module level for Python 3.14t pickling.

    Args:
        ret_dict: Shared dictionary for returning results
        lemma_name: Name of the lemma to prove
        proof_exec_callback: Callback for proof executor
        env_settings: Environment settings
        eval_settings: Evaluation settings
        policy_config: Dictionary containing policy configuration
        proof_dump_file_name: Path to proof dump file
        log_dir: Directory for log files
        theorem_idx: Index of the theorem being proved
        path: File path being processed
    """
    try:
        # Import dependencies inside subprocess
        from copra.main.policy_factory import PolicyFactory
        from itp_interface.tools.log_utils import setup_logger
        import os

        # Create a subprocess-specific logger
        log_file = os.path.join(log_dir, f"eval_thm_{theorem_idx}.log")
        subprocess_logger = setup_logger(
            f"{__name__}.thm_{theorem_idx}",
            log_file,
            logging.INFO,
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        subprocess_logger.info(f"Starting proof attempt for theorem {theorem_idx}: {lemma_name}")

        # Extract configuration
        policy_name = policy_config['policy_name']
        eval_benchmark_language = policy_config['eval_benchmark_language']
        checkpoint_dir = policy_config['checkpoint_dir']
        should_checkpoint = policy_config['should_checkpoint']

        # Recreate policy_prompter inside subprocess (avoids pickling OpenAI client with locks)
        eval_settings = policy_config['eval_settings']
        prompt_settings = policy_config['prompt_settings']
        eval_benchmark = policy_config['eval_benchmark']
        lemma_name_for_prompter = policy_config['lemma_name']

        policy_prompter, informal_proof_repo, informal_proof_dump_directory = \
            PolicyFactory.create_policy_prompter(
                eval_settings, prompt_settings, eval_benchmark, lemma_name_for_prompter, subprocess_logger
            )

        if policy_name == PolicyName.Dfs:
            dfs_tree_search = DFSTreeSearch(language=eval_benchmark_language)
            search_guidance_policy = GptGuidedTreeSearchPolicy(
                checkpoint_dir,
                lemma_name,
                policy_prompter,
                dfs_tree_search,
                checkpoint_on_exit=should_checkpoint,
                language=eval_benchmark_language
            )
        elif policy_name == PolicyName.Hammer:
            dfs_tree_search = DFSTreeSearch(language=eval_benchmark_language)
            search_guidance_policy = GptGuidedTreeSearchPolicy(
                checkpoint_dir,
                lemma_name,
                policy_prompter,
                dfs_tree_search,
                checkpoint_on_exit=should_checkpoint,
                language=eval_benchmark_language
            )
        elif policy_name == PolicyName.FewShot:
            informal_proof_repo = policy_config.get('informal_proof_repo', None)
            search_guidance_policy = FewShotGptPolicy(
                lemma_name,
                checkpoint_dir,
                lemma_name,
                policy_prompter,
                checkpoint_on_exit=should_checkpoint,
                language=eval_benchmark_language,
                logger=subprocess_logger,
                informal_proof_repo=informal_proof_repo
            )
        elif policy_name == PolicyName.InformalFewShot:
            informal_proof_repo = policy_config['informal_proof_repo']
            informal_proof_dump_directory = policy_config['informal_proof_dump_directory']
            search_guidance_policy = InformalFewShotGptPolicy(
                lemma_name,
                checkpoint_dir,
                lemma_name,
                policy_prompter,
                informal_proof_repo,
                checkpoint_on_exit=should_checkpoint,
                language=eval_benchmark_language,
                logger=subprocess_logger,
                informal_proof_dump_dir=informal_proof_dump_directory
            )
        else:
            raise Exception(f"Unknown policy name: {policy_name}")

        with ProofEnv(
            f"basic_proof_env_{lemma_name}",
            proof_exec_callback,
            lemma_name,
            retrieval_strategy=env_settings.retrieval_strategy,
            max_proof_depth=eval_settings.max_proof_depth,
            always_retrieve_thms=eval_settings.always_use_useful_theorem_retrieval,
            logger=subprocess_logger
        ) as env:
            with search_guidance_policy:
                agent = ProofAgent(
                    f"proof_agent_{lemma_name}",
                    search_guidance_policy,
                    eval_settings.should_checkpoint,
                    proof_dump_file_name,
                    logger=subprocess_logger
                )
                agent.run_episodes_till_stop(
                    env,
                    episodes=eval_settings.max_number_of_episodes,
                    render=eval_settings.render,
                    stop_policy=check_query_limit_reached(eval_settings.max_steps_per_episode),
                    policy_info_message=query_limit_info_message(eval_settings.max_steps_per_episode)
                )
            proof_res = env.proof_search_res
            ret_dict["proof_res"] = proof_res
            ret_dict["attempted_success"] = True
            ret_dict["service_down"] = False
    except ServiceDownError:
        subprocess_logger.exception(f"ServiceDownError occurred while proving lemma: {lemma_name} in file {path}")
        ret_dict["attempted_success"] = False
        ret_dict["service_down"] = True
    except Exception:
        subprocess_logger.exception(f"Exception occurred while proving lemma: {lemma_name} in file {path}")
        ret_dict["attempted_success"] = False
        ret_dict["service_down"] = False


class ProofExecutionManager:
    """Manages proof execution with timeout, retry logic, and time budget tracking."""

    def __init__(self, logger: logging.Logger):
        """
        Initialize the proof execution manager.

        Args:
            logger: Logger instance
        """
        self.logger = logger

    def run_proof_with_timeout(
        self,
        lemma_name: str,
        path: str,
        proof_exec_callback: ProofExecutorCallback,
        env_settings: EnvSettings,
        eval_settings: EvalSettings,
        policy_config: Dict[str, Any],
        proof_dump_file_name: str,
        timeout: int,
        log_dir: str,
        theorem_idx: int
    ) -> Dict[str, Any]:
        """
        Run a single proof attempt with timeout.

        Args:
            lemma_name: Name of the lemma to prove
            path: File path being processed
            proof_exec_callback: Callback for proof executor
            env_settings: Environment settings
            eval_settings: Evaluation settings
            policy_config: Policy configuration dictionary
            proof_dump_file_name: Path to proof dump file
            timeout: Timeout in seconds
            log_dir: Directory for log files
            theorem_idx: Index of the theorem being proved

        Returns:
            Dictionary containing proof results and status
        """
        self.logger.info(f"Running the prover agent for lemma: {lemma_name} (theorem_idx={theorem_idx}) with timeout: {timeout} seconds")

        # Get the appropriate executor (free-threading for 3.14t+, multiprocessing for older)
        executor = get_executor()

        # Execute with timeout
        return_dict, elapsed_time = executor.execute_with_timeout(
            target=_run_prover_wrapper,
            args=(
                lemma_name,
                proof_exec_callback,
                env_settings,
                eval_settings,
                policy_config,
                proof_dump_file_name,
                log_dir,
                theorem_idx,
                path
            ),
            timeout=timeout
        )

        result = {
            'return_dict': return_dict if return_dict is not None else {},
            'elapsed_time': elapsed_time
        }

        return result
