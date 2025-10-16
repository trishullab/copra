#!/usr/bin/env python3

"""
Benchmark Evaluation Entry Point

This module provides the main entry point for running proof benchmark evaluations.
It orchestrates lemma discovery, policy creation, and proof execution while
maintaining the original public API for backwards compatibility.
"""

import hydra
import copy
import logging
import os
import random
import time
import math
import typing

from copra.gpts.llama_access import LlamaAccess, ServiceDownError
from copra.main.config import (
    EnvSettings, EvalBenchmark, EvalDataset, EvalProofResults, EvalSettings,
    Experiments, EvalRunCheckpointInfo, PromptSettings, parse_config
)
from copra.main.policy_factory import PolicyFactory
from copra.main.lemma_discovery import discover_lemmas_with_timeout
from copra.main.proof_execution import ProofExecutionManager
from copra.main.checkpoint_manager import CheckpointManager
from copra.main.parallel_theorem_execution import ParallelTheoremExecutor
from copra.tools.misc import model_supports_openai_api
from itp_interface.tools.log_utils import setup_logger
from itp_interface.rl.proof_tree import ProofSearchResult
from itp_interface.rl.proof_action import ProofAction
from itp_interface.tools.isabelle_executor import IsabelleExecutor
from itp_interface.tools.proof_exec_callback import ProofExecutorCallback


def _initialize_services(
    eval_settings: EvalSettings,
    eval_benchmark: EvalBenchmark,
    eval_checkpoint_info: EvalRunCheckpointInfo,
    logger: logging.Logger
) -> None:
    """
    Initialize required services (Llama, Isabelle) based on configuration.

    Args:
        eval_settings: Evaluation settings
        eval_benchmark: Benchmark configuration
        eval_checkpoint_info: Checkpoint information
        logger: Logger instance
    """
    # Initialize Llama service if using non-OpenAI model
    if eval_settings.gpt_model_name is not None and \
       len(eval_settings.gpt_model_name) != 0 and \
       not model_supports_openai_api(eval_settings.gpt_model_name):
        llama_logger = setup_logger(
            __name__ + "_llama",
            os.path.join(eval_checkpoint_info.logging_dirs[-1], "llama.log"),
            logging.INFO,
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        LlamaAccess.class_init(eval_settings.gpt_model_name, eval_settings.temperature,
                              debug=False, logger=llama_logger)

    # Initialize Isabelle service if needed
    if eval_benchmark.language == ProofAction.Language.ISABELLE:
        isabelle_logger = setup_logger(
            __name__ + "_isabelle",
            os.path.join(eval_checkpoint_info.logging_dirs[-1], "isabelle.log"),
            logging.INFO,
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        # Check if environment variable PISA_PORT is set
        if "PISA_PORT" not in os.environ:
            os.environ["PISA_PORT"] = "17000"
            if IsabelleExecutor.check_server_running(isabelle_logger):
                raise Exception(
                    "PISA_PORT environment variable is not set but the PISA service is "
                    "already running on default port 17000. Please set the PISA_PORT "
                    "environment variable to the port on which the PISA service is running."
                )
        IsabelleExecutor.start_server(isabelle_logger)


def _shutdown_services(
    eval_settings: EvalSettings,
    eval_benchmark: EvalBenchmark
) -> None:
    """
    Shutdown services that were initialized.

    Args:
        eval_settings: Evaluation settings
        eval_benchmark: Benchmark configuration
    """
    # Shutdown Llama service if it was initialized
    if eval_settings.gpt_model_name is not None and \
       len(eval_settings.gpt_model_name) != 0 and \
       not model_supports_openai_api(eval_settings.gpt_model_name):
        LlamaAccess.class_kill()

    # Shutdown Isabelle service if it was initialized
    if eval_benchmark.language == ProofAction.Language.ISABELLE:
        IsabelleExecutor.stop_server()


def _manage_isabelle_server(
    server_use_count: int,
    max_server_use_count: int,
    eval_benchmark: EvalBenchmark,
    logger: logging.Logger
) -> int:
    """
    Manage Isabelle server restarts and health checks.

    Args:
        server_use_count: Current server use count
        max_server_use_count: Maximum uses before restart
        eval_benchmark: Benchmark configuration
        logger: Logger instance

    Returns:
        Updated server use count (0 if restarted, incremented otherwise)
    """
    if eval_benchmark.language != ProofAction.Language.ISABELLE:
        return server_use_count + 1

    # Restart server if use count exceeded
    if server_use_count >= max_server_use_count:
        logger.warning(f"Server use count exceeded {max_server_use_count}. Restarting the PISA service.")
        IsabelleExecutor.stop_server()
        logger.warning("Stopped the PISA service.")
        logger.warning("Waiting for 10 seconds before starting the PISA service.")
        time.sleep(15)
        logger.warning("Starting the PISA service again.")
        IsabelleExecutor.start_server(logger)
        logger.warning("Started the PISA service.")
        return 1

    # Check if server is down and restart if needed
    if not IsabelleExecutor.check_server_running(logger):
        try:
            IsabelleExecutor.stop_server()
        except Exception:
            pass
        logger.warning("PISA service is down. Restarting it.")
        IsabelleExecutor.start_server(logger)
        logger.warning("Restarted the PISA service.")

    return server_use_count + 1


def _filter_lemmas(
    file,
    all_lemmas: typing.List[str],
    eval_settings: EvalSettings,
    path: str,
    logger: logging.Logger
) -> typing.List[str]:
    """
    Filter and optionally sample lemmas based on file configuration.

    Args:
        file: File configuration object
        all_lemmas: List of all discovered lemmas
        eval_settings: Evaluation settings
        path: File path
        logger: Logger instance

    Returns:
        Filtered and optionally sampled list of lemmas
    """
    # Determine which lemmas to prove
    if isinstance(file.theorems, str) and file.theorems == "*":
        file.theorems = list(all_lemmas)
        file.theorems.sort()  # Sort to ensure consistent order
    elif isinstance(file.theorems, list):
        # Keep only lemmas that exist in the file
        intersection = set(file.theorems).intersection(all_lemmas)
        file.theorems = [x for x in file.theorems if x in intersection]
    else:
        raise ValueError(f"Invalid theorems: {file.theorems}")

    logger.info(f"Discovered {len(file.theorems)} lemmas to prove in {path}")
    logger.info(f"Lemmas to prove in file {path}: \n{file.theorems}")

    # Sample if requested
    if eval_settings.sample < 1.0:
        sample_size = math.ceil(len(file.theorems) * eval_settings.sample)
        logger.info(f"Sampling {sample_size} lemmas from {len(file.theorems)} lemmas in file {path}")
        random.seed(eval_settings.sample_seed)
        file.theorems = list(random.sample(file.theorems, sample_size))
        logger.info(f"Sampled lemmas to prove in file {path}: \n{file.theorems}")

    return file.theorems


def _create_proof_dump_file(path: str, dataset: EvalDataset, eval_settings: EvalSettings) -> str:
    """
    Create proof dump file and return its path.

    Args:
        path: File path
        dataset: Dataset configuration
        eval_settings: Evaluation settings

    Returns:
        Path to proof dump file
    """
    proof_dump_file_name = os.path.join(
        eval_settings.proof_dump_dir,
        f"{path.replace('/', '_')}.txt"
    )

    if not os.path.exists(proof_dump_file_name):
        with open(proof_dump_file_name, "w") as f:
            f.write(f"File: {path}\n")
            f.write(f"Dataset:\n {dataset.to_json(indent=4)}\n")
            f.write(f"Evaluation Settings:\n {eval_settings.to_json(indent=4)}\n")

    return proof_dump_file_name


def _handle_proof_timeout(
    lemma_name: str,
    path: str,
    elapsed_time: float,
    no_proof_res: ProofSearchResult,
    checkpoint_manager: CheckpointManager,
    attempt_idx: int
) -> None:
    """
    Handle proof timeout case by saving appropriate result.

    Args:
        lemma_name: Lemma name
        path: File path
        elapsed_time: Time elapsed
        no_proof_res: Empty proof result template
        checkpoint_manager: Checkpoint manager
        attempt_idx: Current attempt index
    """
    checkpoint_manager.logger.info(
        f"Prover Agent for lemma: {lemma_name} in file {path} got killed as it timed out."
    )

    proof_res_queries = checkpoint_manager.get_previous_queries(path, lemma_name)
    proof_attempt_idx = checkpoint_manager.get_previous_attempt_idx(path, lemma_name, attempt_idx)

    proof_res_chkpt = copy.deepcopy(no_proof_res)
    proof_res_chkpt.is_timeout = True
    proof_res_chkpt.proof_time_in_secs = elapsed_time
    proof_res_chkpt.additional_info["attempt_idx"] = proof_attempt_idx
    proof_res_chkpt.additional_info["total_queries"] = proof_res_queries

    checkpoint_manager.save_proof_result(path, lemma_name, proof_res_chkpt, success=False)


def _handle_proof_success(
    lemma_name: str,
    path: str,
    return_dict: typing.Dict[str, typing.Any],
    checkpoint_manager: CheckpointManager,
    attempt_idx: int,
    eval_settings: EvalSettings
) -> None:
    """
    Handle successful proof completion.

    Args:
        lemma_name: Lemma name
        path: File path
        return_dict: Return dictionary from proof execution
        checkpoint_manager: Checkpoint manager
        attempt_idx: Current attempt index
        eval_settings: Evaluation settings
    """
    proof_res_queries = checkpoint_manager.get_previous_queries(path, lemma_name)
    proof_attempt_idx = checkpoint_manager.get_previous_attempt_idx(path, lemma_name, attempt_idx)

    proof_res_chkpt: ProofSearchResult = return_dict["proof_res"]
    proof_res_chkpt.additional_info["attempt_idx"] = proof_attempt_idx
    proof_res_chkpt.additional_info["total_queries"] = \
        proof_res_queries + proof_res_chkpt.additional_info["queries"]

    if not proof_res_chkpt.proof_found and "queries" in proof_res_chkpt.additional_info:
        proof_res_chkpt.is_inference_exhausted = \
            proof_res_chkpt.additional_info["queries"] >= eval_settings.max_steps_per_episode

    checkpoint_manager.save_proof_result(path, lemma_name, proof_res_chkpt, success=True)


def _process_lemma(
    lemma_name: str,
    path: str,
    file,
    env_settings: EnvSettings,
    eval_settings: EvalSettings,
    eval_benchmark: EvalBenchmark,
    prompt_settings: PromptSettings,
    proof_exec_callback: ProofExecutorCallback,
    proof_dump_file_name: str,
    checkpoint_manager: CheckpointManager,
    proof_executor: ProofExecutionManager,
    attempt_idx: int,
    track_time: bool,
    time_budget_tracker: typing.Dict[str, typing.Dict[str, float]],
    log_dir: str,
    theorem_idx: int,
    logger: logging.Logger
) -> bool:
    """
    Process a single lemma proof attempt.

    Args:
        lemma_name: Name of the lemma to prove
        path: File path
        file: File configuration
        env_settings: Environment settings
        eval_settings: Evaluation settings
        eval_benchmark: Benchmark configuration
        prompt_settings: Prompt settings
        proof_exec_callback: Proof executor callback
        proof_dump_file_name: Path to proof dump file
        checkpoint_manager: Checkpoint manager
        proof_executor: Proof execution manager
        attempt_idx: Current attempt index
        track_time: Whether to track time budget
        time_budget_tracker: Time budget tracker dictionary
        log_dir: Directory for log files
        theorem_idx: Index of the theorem being proved
        logger: Logger instance

    Returns:
        True if proof was attempted, False otherwise
    """
    # Initialize time budget if needed
    if track_time and lemma_name not in time_budget_tracker[path]:
        time_budget_tracker[path][lemma_name] = eval_benchmark.timeout_per_theorem_in_secs

    if track_time and time_budget_tracker[path][lemma_name] <= 0:
        logger.info(f"Time budget exhausted for lemma: {lemma_name} in file {path} so skipping it.")
        return False

    logger.info(f"Attempting to prove lemma: {lemma_name}")

    # Create policy configuration
    try:
        # We don't create the policy_prompter here anymore - it will be created
        # inside the subprocess to avoid pickling issues with OpenAI client (which has thread locks)
        policy_config = PolicyFactory.create_policy_config(
            eval_settings, eval_benchmark, prompt_settings, lemma_name
        )
    except Exception:
        logger.exception(f"Failed to create policy for lemma: {lemma_name}")
        return False

    # Create empty proof result template
    no_proof_res = ProofSearchResult(
        None, False, lemma_name, [], -1, -1,
        possible_failed_paths=-1, num_of_backtracks=-1,
        is_timeout=False, is_inference_exhausted=False,
        longest_success_path=-1, additional_info={},
        language=eval_benchmark.language
    )

    # Check if we should attempt this proof
    max_retry_attempts = file.max_retry_attempts_limits.get(lemma_name, eval_settings.proof_retries)
    if not checkpoint_manager.should_attempt_proof(path, lemma_name, attempt_idx, max_retry_attempts):
        return False

    # Run proof with retries
    should_retry = True
    max_retry = 4  # Retry only for service failures
    logger.info(f"Attempt {attempt_idx + 1} for proving lemma: {lemma_name} in file {path}")

    while should_retry and max_retry > 0:
        # Calculate timeout
        timeout = eval_benchmark.timeout_per_theorem_in_secs
        if track_time and time_budget_tracker[path][lemma_name] < timeout:
            timeout = time_budget_tracker[path][lemma_name]

        # Run proof
        try:
            result = proof_executor.run_proof_with_timeout(
                lemma_name, path, proof_exec_callback, env_settings,
                eval_settings, policy_config, proof_dump_file_name, timeout,
                log_dir, theorem_idx
            )
        except Exception:
            logger.exception(f"Exception occurred while proving lemma: {lemma_name} in file {path}")
            _handle_proof_timeout(lemma_name, path, 0, no_proof_res, checkpoint_manager, attempt_idx)
            return True

        return_dict = result['return_dict']
        elapsed_time = result['elapsed_time']

        # Handle Isabelle server down case
        if eval_benchmark.language == ProofAction.Language.ISABELLE and \
           not IsabelleExecutor.check_server_running(logger) and \
           "attempted_success" in return_dict and \
           not return_dict["attempted_success"]:
            logger.warning("PISA service is down. The proof might have failed because the server was down.")
            return_dict.pop("attempted_success")

        # Update time budget
        if track_time:
            time_budget_tracker[path][lemma_name] -= elapsed_time

        # Handle different result cases
        if track_time and time_budget_tracker[path][lemma_name] <= 0:
            logger.info(f"Time budget exhausted for lemma: {lemma_name} in file {path}")
            _handle_proof_timeout(lemma_name, path, elapsed_time, no_proof_res, checkpoint_manager, attempt_idx)
            logger.info(f"Dumping proof search result:\nProof FAILED (Time budget exhausted) for lemma: {lemma_name}\n")
            should_retry = False

        elif "attempted_success" not in return_dict:
            # Timeout case
            _handle_proof_timeout(lemma_name, path, elapsed_time, no_proof_res, checkpoint_manager, attempt_idx)
            logger.info(f"Dumping proof search result:\nProof FAILED (Timeout) for lemma: {lemma_name}\n")
            should_retry = False

        elif not return_dict["attempted_success"]:
            # Failure case - check if we should retry for service issues
            if not return_dict.get("service_down", False) or \
               (eval_settings.gpt_model_name is not None and
                len(eval_settings.gpt_model_name) != 0 and
                model_supports_openai_api(eval_settings.gpt_model_name)) or \
               max_retry <= 1:
                _handle_proof_timeout(lemma_name, path, elapsed_time, no_proof_res, checkpoint_manager, attempt_idx)
                logger.info(f"Dumping proof search result:\nProof FAILED for lemma: {lemma_name}\n")
                should_retry = False
            elif return_dict.get("service_down", False):
                # Retry for service down
                should_retry = True
                logger.info("Killing the llama process")
                LlamaAccess.class_kill()
                logger.info("Killed the llama process")
                logger.info("Restarting the llama process")
                # Get llama logger from logs
                llama_logger = logging.getLogger(__name__ + "_llama")
                LlamaAccess.class_init(eval_settings.gpt_model_name,
                                      eval_settings.temperature, debug=False, logger=llama_logger)
                logger.info("Restarted the llama process")

        else:
            # Success case
            _handle_proof_success(lemma_name, path, return_dict, checkpoint_manager, attempt_idx, eval_settings)
            # Log the final proof result to main log
            proof_res: ProofSearchResult = return_dict["proof_res"]
            logger.info(f"Dumping proof search result:\n{proof_res}\n")
            should_retry = False

        max_retry -= 1

    return True


def _process_file(
    file,
    dataset: EvalDataset,
    env_settings: EnvSettings,
    eval_settings: EvalSettings,
    eval_benchmark: EvalBenchmark,
    prompt_settings: PromptSettings,
    checkpoint_manager: CheckpointManager,
    proof_executor: ProofExecutionManager,
    attempt_idx: int,
    track_time: bool,
    time_budget_tracker: typing.Dict[str, typing.Dict[str, float]],
    skip_files_in_checkpoint: bool,
    server_use_count: int,
    max_server_use_count: int,
    log_dir: str,
    logger: logging.Logger,
    parallel_executor: typing.Optional[ParallelTheoremExecutor] = None
) -> typing.Tuple[bool, int]:
    """
    Process a single file from the dataset.

    Args:
        file: File configuration
        dataset: Dataset configuration
        env_settings: Environment settings
        eval_settings: Evaluation settings
        eval_benchmark: Benchmark configuration
        prompt_settings: Prompt settings
        checkpoint_manager: Checkpoint manager
        proof_executor: Proof execution manager
        attempt_idx: Current attempt index
        track_time: Whether to track time budget
        time_budget_tracker: Time budget tracker
        skip_files_in_checkpoint: Whether to skip checkpointed files
        server_use_count: Current server use count
        max_server_use_count: Maximum server use count
        log_dir: Directory for log files
        logger: Logger instance
        parallel_executor: Optional parallel theorem executor for running theorems in parallel

    Returns:
        Tuple of (any_proof_attempted, updated_server_use_count)
    """
    path = os.path.join(dataset.project, file.path)
    any_proof_attempted = False

    # Initialize time budget for this file
    if track_time and path not in time_budget_tracker:
        if len(file.max_time_limits_in_secs) > 0:
            time_budget_tracker[path] = copy.deepcopy(file.max_time_limits_in_secs)
        else:
            time_budget_tracker[path] = {}

    # Check if file should be skipped
    if checkpoint_manager.should_skip_file(path, skip_files_in_checkpoint):
        return any_proof_attempted, server_use_count

    # Create proof dump file
    proof_dump_file_name = _create_proof_dump_file(path, dataset, eval_settings)

    # Add path to checkpoint maps
    checkpoint_manager.add_path_to_maps(path)

    # Create proof executor callbacks
    proof_exec_callback = ProofExecutorCallback(
        project_folder=dataset.project,
        file_path=path,
        language=eval_benchmark.language,
        use_hammer=False
        if eval_benchmark.language == ProofAction.Language.LEAN4 or
        eval_benchmark.language == ProofAction.Language.LEAN
        else eval_settings.use_hammer,
        timeout_in_secs=eval_settings.timeout_in_secs,
        use_human_readable_proof_context=eval_settings.use_human_readable_proof_context,
        suppress_error_log=True,
        always_use_retrieval=eval_settings.always_use_useful_theorem_retrieval,
        logger=logger
    )

    lemma_discovery_callback = ProofExecutorCallback(
        project_folder=dataset.project,
        file_path=path,
        language=eval_benchmark.language,
        use_hammer=False,
        timeout_in_secs=eval_settings.timeout_in_secs,
        use_human_readable_proof_context=eval_settings.use_human_readable_proof_context,
        suppress_error_log=True,
        always_use_retrieval=False,
        logger=logger
    )

    # Manage server health
    server_use_count = _manage_isabelle_server(
        server_use_count, max_server_use_count, eval_benchmark, logger
    )

    # Discover lemmas
    file_timeout = min(3000, eval_settings.timeout_in_secs * eval_settings.max_proof_depth * 50)
    lemmas_to_prove = discover_lemmas_with_timeout(path, lemma_discovery_callback, file_timeout, logger)

    if lemmas_to_prove is None:
        return any_proof_attempted, server_use_count

    # Filter and sample lemmas
    lemmas = _filter_lemmas(file, lemmas_to_prove, eval_settings, path, logger)

    # Process each lemma - either in parallel or sequentially
    if parallel_executor is not None:
        # Execute theorems in parallel
        any_proof_attempted = parallel_executor.execute_theorems_in_parallel(
            lemmas, path, file, env_settings, eval_settings,
            eval_benchmark, prompt_settings, proof_exec_callback,
            proof_dump_file_name, checkpoint_manager, proof_executor,
            attempt_idx, track_time, time_budget_tracker, log_dir,
            _process_lemma, logger
        )
    else:
        # Execute theorems sequentially (original behavior)
        for theorem_idx, lemma_name in enumerate(lemmas):
            try:
                attempted = _process_lemma(
                    lemma_name, path, file, env_settings, eval_settings,
                    eval_benchmark, prompt_settings, proof_exec_callback,
                    proof_dump_file_name, checkpoint_manager, proof_executor,
                    attempt_idx, track_time, time_budget_tracker, log_dir,
                    theorem_idx, logger
                )
                any_proof_attempted = any_proof_attempted or attempted
            except Exception:
                logger.exception(f"Exception occurred while proving lemma: {lemma_name} in file {path}")

    return any_proof_attempted, server_use_count


def eval_dataset(
    env_settings: EnvSettings,
    eval_benchmark: EvalBenchmark,
    prompt_settings: PromptSettings,
    dataset: EvalDataset,
    eval_settings: EvalSettings,
    eval_checkpoint_info: EvalRunCheckpointInfo,
    eval_proof_results: EvalProofResults,
    logger: logging.Logger = None
) -> None:
    """
    Evaluate a dataset by attempting to prove theorems.

    This function maintains the original API for backwards compatibility.

    Args:
        env_settings: Environment settings
        eval_benchmark: Benchmark configuration
        prompt_settings: Prompt settings
        dataset: Dataset to evaluate
        eval_settings: Evaluation settings
        eval_checkpoint_info: Checkpoint information
        eval_proof_results: Proof results
        logger: Optional logger instance
    """
    logger = logger or logging.getLogger(__name__)

    # Initialize services
    _initialize_services(eval_settings, eval_benchmark, eval_checkpoint_info, logger)

    try:
        # Setup configuration
        skip_files_in_checkpoint = os.environ.get("SKIP_FILES_IN_CHECKPOINT", "False").lower() == "true"

        if eval_settings.proof_retries > 1:
            assert eval_settings.temperature > 0.0, "Proof retries is only supported for temperature > 0.0"

        track_time = os.environ.get("STRICT_TIME_BUDGET_ACCROSS_ATTEMPTS", "False").lower() == "true"
        if track_time:
            logger.info(
                f"Strict time budget across attempts is enabled. Proofs will not be attempted "
                f"beyond {eval_benchmark.timeout_per_theorem_in_secs} seconds."
            )

        # Check if parallel theorem execution is enabled
        enable_parallel_theorems = os.environ.get("ENABLE_PARALLEL_THEOREMS", "False").lower() == "true"
        max_parallel_workers = None
        if os.environ.get("MAX_PARALLEL_WORKERS"):
            try:
                max_parallel_workers = int(os.environ.get("MAX_PARALLEL_WORKERS"))
            except ValueError:
                logger.warning(f"Invalid MAX_PARALLEL_WORKERS value, using default")

        time_budget_tracker = {}
        server_use_count = 0
        max_server_use_count = 5

        # Create managers
        checkpoint_manager = CheckpointManager(eval_checkpoint_info, eval_proof_results, logger)
        proof_executor = ProofExecutionManager(logger)

        # Create parallel theorem executor if enabled
        parallel_executor = None
        if enable_parallel_theorems:
            parallel_executor = ParallelTheoremExecutor(logger, max_workers=max_parallel_workers)
            logger.info("Parallel theorem execution is ENABLED")
        else:
            logger.info("Parallel theorem execution is DISABLED (sequential execution)")

        # Main evaluation loop
        proof_attempts_done = False
        for attempt_idx in range(eval_settings.proof_retries):
            if proof_attempts_done:
                break

            any_proof_attempted = False

            for file in dataset.files:
                # Get log directory from checkpoint info
                log_dir = eval_checkpoint_info.logging_dirs[-1]
                attempted, server_use_count = _process_file(
                    file, dataset, env_settings, eval_settings, eval_benchmark,
                    prompt_settings, checkpoint_manager, proof_executor,
                    attempt_idx, track_time, time_budget_tracker,
                    skip_files_in_checkpoint, server_use_count,
                    max_server_use_count, log_dir, logger, parallel_executor
                )
                any_proof_attempted = any_proof_attempted or attempted

            proof_attempts_done = not any_proof_attempted

    finally:
        # Shutdown services
        _shutdown_services(eval_settings, eval_benchmark)


def measure_success(
    benchmark: EvalBenchmark,
    eval_settings: EvalSettings,
    eval_proof_results: EvalProofResults,
    logger: logging.Logger = None
) -> None:
    """
    Measure and log success rate for the benchmark.

    Args:
        benchmark: Benchmark configuration
        eval_settings: Evaluation settings
        eval_proof_results: Proof results
        logger: Optional logger instance
    """
    logger = logger or logging.getLogger(__name__)
    success_count = 0
    proofs_dump_file = os.path.join(eval_settings.proof_dump_dir, "benchmark_proof_results.txt")
    proof_dump_file_exists = os.path.exists(proofs_dump_file)
    open_mode = "a" if proof_dump_file_exists else "w"

    with open(proofs_dump_file, open_mode) as f:
        if not proof_dump_file_exists:
            f.write(f"Settings: \n{eval_settings.to_json(indent=4)}\n")
            f.write(f"Benchmark: \n{benchmark.to_json(indent=4)}\n")

        for path, proofs in eval_proof_results.theorem_map.items():
            for lemma_name, proof_res in proofs.items():
                if proof_res.proof_found:
                    success_count += 1
                    logger.info(f"Proof found for lemma: {lemma_name} in file {path}")
                else:
                    logger.info(f"Proof not found for lemma: {lemma_name} in file {path}")
                f.write(f"Lemma: {lemma_name}\n")
                f.write(f"File: {path}\n")
                f.write(f"Proof/Incomplete proof: \n{proof_res}\n")

        total_attempted = sum([len(x) for _, x in eval_proof_results.theorem_map.items()])
        success_rate = success_count / total_attempted if total_attempted > 0 else 0
        logger.info(f"Success rate: {success_count}/{total_attempted} = {success_rate} for benchmark: {benchmark.name}")
        f.write(f"Success rate: {success_count}/{total_attempted} = {success_rate} for benchmark: {benchmark.name}\n")


def eval_benchmark(
    experiment: Experiments,
    log_dir: str,
    logger: logging.Logger = None,
    timestr: str = None
) -> None:
    """
    Run benchmark evaluation for an experiment.

    Args:
        experiment: Experiment configuration
        log_dir: Directory for logs
        logger: Optional logger instance
        timestr: Optional timestamp string
    """
    trial_cnt = 1
    env_settings = experiment.env_settings
    eval_settings = experiment.eval_settings
    benchmark = experiment.benchmark
    checkpoint_dir = experiment.eval_settings.checkpoint_dir
    prompt_settings = experiment.prompt_settings

    eval_settings.checkpoint_dir = os.path.join(
        checkpoint_dir, benchmark.name, eval_settings.name, prompt_settings.name
    )
    os.makedirs(eval_settings.checkpoint_dir, exist_ok=True)

    # Load checkpoint file if it exists
    checkpoint_file = os.path.join(eval_settings.checkpoint_dir, "checkpoint_info.json")
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, "r") as f:
            checkpoint_info = EvalRunCheckpointInfo.from_json(f.read())
        eval_settings.proof_dump_dir = checkpoint_info.proof_dump_dir
        checkpoint_info.logging_dirs.append(log_dir)
    else:
        time_now = time.strftime("%Y%m%d-%H%M%S") if timestr is None else timestr
        eval_settings.proof_dump_dir = os.path.join(eval_settings.proof_dump_dir, benchmark.name, time_now)
        os.makedirs(eval_settings.proof_dump_dir, exist_ok=True)
        checkpoint_info = EvalRunCheckpointInfo(
            checkpoint_file=checkpoint_file,
            proof_dump_dir=eval_settings.proof_dump_dir,
            logging_dirs=[log_dir],
            theorem_maps={}
        )

    # Load proof results file if it exists
    eval_proof_file = os.path.join(eval_settings.proof_dump_dir, "proof_results.json")
    if os.path.exists(eval_proof_file):
        with open(eval_proof_file, "r") as f:
            eval_proof_results = EvalProofResults.from_json(f.read())
    else:
        eval_proof_results = EvalProofResults(
            path=os.path.join(eval_settings.proof_dump_dir, "proof_results.json"),
            theorem_map={}
        )

    while trial_cnt > 0:
        try:
            logger = logger or logging.getLogger(__name__)
            for dataset in benchmark.datasets:
                eval_dataset(
                    env_settings, benchmark, prompt_settings, dataset,
                    eval_settings, checkpoint_info, eval_proof_results, logger=logger
                )
            measure_success(benchmark, eval_settings, eval_proof_results, logger=logger)
            trial_cnt = 0
        except Exception:
            trial_cnt -= 1
            logger.exception(f"Exception occurred. Retrying {trial_cnt} more times.")
            time.sleep(10)

    logger.info(f"Finished running experiment: \n{experiment.to_json(indent=4)}")


@hydra.main(config_path="config", config_name="experiments", version_base="1.2")
def main(cfg):
    """
    Main entry point for benchmark evaluation.

    Args:
        cfg: Hydra configuration
    """
    experiment = parse_config(cfg)
    timestr = time.strftime("%Y%m%d-%H%M%S")
    log_dir = ".log/evals/benchmark/{}/{}".format(experiment.benchmark.name, timestr)
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "eval.log")
    logger = setup_logger(
        __name__, log_path, logging.INFO,
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger.info(f"Pid: {os.getpid()}")
    logger.info(f"Running Experiment: {experiment.to_json(indent=4)}")
    eval_benchmark(experiment, log_dir, logger=logger, timestr=timestr)


if __name__ == "__main__":
    main()
