#!/usr/bin/env python3

"""
Parallel Theorem Execution Module

This module provides functionality to execute proof search for multiple theorems in parallel,
using either multiprocessing (Python < 3.14) or threading (Python 3.14t+).
"""

import logging
import typing
import sys
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from typing import Dict, Any, List, Tuple, Optional

from copra.main.config import EnvSettings, EvalSettings, EvalBenchmark, PromptSettings
from copra.main.proof_execution import ProofExecutionManager
from copra.main.checkpoint_manager import CheckpointManager
from itp_interface.tools.proof_exec_callback import ProofExecutorCallback


def _get_max_workers(num_theorems: int, max_workers: Optional[int] = None) -> int:
    """
    Determine the optimal number of worker processes/threads.

    Args:
        num_theorems: Total number of theorems to prove
        max_workers: Optional maximum number of workers

    Returns:
        Optimal number of workers
    """
    import os

    if max_workers is not None:
        return min(max_workers, num_theorems)

    # Get CPU count
    cpu_count = os.cpu_count() or 1

    # Use at most half the available CPUs to avoid overloading the system
    # but at least 1 and at most the number of theorems
    return min(max(1, cpu_count // 2), num_theorems)


def _should_use_threading() -> bool:
    """
    Determine whether to use threading (3.14t+) or multiprocessing (< 3.14).

    Returns:
        True if threading should be used, False if multiprocessing
    """
    if sys.version_info >= (3, 14):
        # Check if free-threading is actually enabled
        try:
            if hasattr(sys, '_is_gil_enabled'):
                gil_enabled = sys._is_gil_enabled()
                if not gil_enabled:
                    # GIL is disabled, use threading
                    return True
        except:
            pass
        # For Python 3.14+, use threading even with GIL enabled
        return True
    else:
        # Use multiprocessing for older Python versions
        return False


class ParallelTheoremExecutor:
    """Executes proof search for multiple theorems in parallel."""

    def __init__(self, logger: logging.Logger, max_workers: Optional[int] = None):
        """
        Initialize the parallel theorem executor.

        Args:
            logger: Logger instance
            max_workers: Optional maximum number of parallel workers
        """
        self.logger = logger
        self.max_workers = max_workers
        self.use_threading = _should_use_threading()

        execution_mode = "threading (Python 3.14t+)" if self.use_threading else "multiprocessing (Python < 3.14)"
        self.logger.info(f"ParallelTheoremExecutor initialized with {execution_mode}")

    def execute_theorems_in_parallel(
        self,
        lemmas: List[str],
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
        time_budget_tracker: Dict[str, Dict[str, float]],
        log_dir: str,
        process_lemma_func: typing.Callable,
        logger: Optional['logging.Logger'] = None
    ) -> bool:
        """
        Execute proof search for multiple theorems in parallel.

        Args:
            lemmas: List of lemma names to prove
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
            process_lemma_func: Function to process a single lemma
            logger: Logger instance (defaults to self.logger if not provided)

        Returns:
            True if any proof was attempted, False otherwise
        """
        # Use provided logger or fall back to self.logger
        if logger is None:
            logger = self.logger
        if len(lemmas) == 0:
            return False

        num_workers = _get_max_workers(len(lemmas), self.max_workers)
        self.logger.info(f"Executing {len(lemmas)} theorems in parallel using {num_workers} workers")

        any_proof_attempted = False

        # Choose executor based on Python version
        if self.use_threading:
            executor_class = ThreadPoolExecutor
        else:
            executor_class = ProcessPoolExecutor

        # Create tasks for each lemma
        with executor_class(max_workers=num_workers) as executor:
            # Submit all tasks
            future_to_lemma = {}
            for theorem_idx, lemma_name in enumerate(lemmas):
                future = executor.submit(
                    process_lemma_func,
                    lemma_name,
                    path,
                    file,
                    env_settings,
                    eval_settings,
                    eval_benchmark,
                    prompt_settings,
                    proof_exec_callback,
                    proof_dump_file_name,
                    checkpoint_manager,
                    proof_executor,
                    attempt_idx,
                    track_time,
                    time_budget_tracker,
                    log_dir,
                    theorem_idx,
                    logger  # Add logger as the last argument
                )
                future_to_lemma[future] = lemma_name

            # Wait for all tasks to complete
            for future in as_completed(future_to_lemma):
                lemma_name = future_to_lemma[future]
                try:
                    attempted = future.result()
                    any_proof_attempted = any_proof_attempted or attempted
                except Exception:
                    logger.exception(f"Exception occurred while proving lemma: {lemma_name} in file {path}")

        return any_proof_attempted
