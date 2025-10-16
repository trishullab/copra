#!/usr/bin/env python3

"""
Lemma Discovery Module

This module provides functionality for discovering lemmas in proof files,
including multiprocessing support for handling timeouts.
"""

import logging
from typing import List, Optional, Dict, Any

from copra.main.parallel_execution import get_executor
from itp_interface.rl.proof_action import ProofAction
from itp_interface.tools.proof_exec_callback import ProofExecutorCallback
from itp_interface.tools.dynamic_coq_proof_exec import DynamicProofExecutor as DynamicCoqProofExecutor
from itp_interface.tools.dynamic_lean_proof_exec import DynamicProofExecutor as DynamicLeanProofExecutor
from itp_interface.tools.dynamic_isabelle_proof_exec import DynamicProofExecutor as DynamicIsabelleProofExecutor
from itp_interface.tools.lean4_sync_executor import (
    get_all_theorems_in_file as get_all_theorems_lean4,
    get_fully_qualified_theorem_name as get_fully_qualified_theorem_name_lean4
)


def get_all_lemmas(proof_exec_callback: ProofExecutorCallback, logger: logging.Logger) -> List[str]:
    """
    Discover all lemmas in a proof file.

    Args:
        proof_exec_callback: Callback for proof executor
        logger: Logger instance

    Returns:
        List of lemma names found in the file
    """
    lemmas_to_prove = []

    if proof_exec_callback.language == ProofAction.Language.LEAN4:
        theorem_details = get_all_theorems_lean4(proof_exec_callback.file_path)
        lemmas_to_prove = [
            get_fully_qualified_theorem_name_lean4(theorem)
            for theorem in theorem_details
        ]
        logger.info(f"Discovered {len(lemmas_to_prove)} lemmas")
        _log_lemma_names(lemmas_to_prove, logger)
        return lemmas_to_prove

    with proof_exec_callback.get_proof_executor() as main_executor:
        if isinstance(main_executor, DynamicLeanProofExecutor):
            lemmas_to_prove = _discover_lean_lemmas(main_executor)
        elif isinstance(main_executor, DynamicCoqProofExecutor):
            lemmas_to_prove = _discover_coq_lemmas(main_executor, logger)
        elif isinstance(main_executor, DynamicIsabelleProofExecutor):
            lemmas_to_prove = _discover_isabelle_lemmas(main_executor, logger)

    logger.info(f"Discovered {len(lemmas_to_prove)} lemmas")
    _log_lemma_names(lemmas_to_prove, logger)
    return lemmas_to_prove


def _discover_lean_lemmas(executor: DynamicLeanProofExecutor) -> List[str]:
    """Discover lemmas in Lean files."""
    executor.run_all_without_exec()
    return executor.find_all_theorems_names()


def _discover_coq_lemmas(executor: DynamicCoqProofExecutor, logger: logging.Logger) -> List[str]:
    """Discover lemmas in Coq files."""
    lemmas_to_prove = []
    while not executor.execution_complete:
        assert not executor.is_in_proof_mode(), "main_executor must not be in proof mode"
        _ = list(executor.run_till_next_lemma_return_exec_stmt())

        if executor.execution_complete:
            break

        lemma_name = executor.get_lemma_name_if_running()
        if lemma_name is None:
            _ = list(executor.run_to_finish_lemma_return_exec())
            if executor.execution_complete:
                break
        else:
            logger.info(f"Discovered lemma: {lemma_name}")
            lemmas_to_prove.append(lemma_name)
            executor.run_to_finish_lemma()

    return lemmas_to_prove


def _discover_isabelle_lemmas(
    executor: DynamicIsabelleProofExecutor,
    logger: logging.Logger
) -> List[str]:
    """Discover lemmas in Isabelle files."""
    lemmas_to_prove = []
    while not executor.execution_complete:
        assert not executor.is_in_proof_mode(), "main_executor must not be in proof mode"
        _ = list(executor.run_till_next_lemma_return_exec_stmt())

        if executor.execution_complete:
            break

        lemma_name = executor.get_lemma_name_if_running()
        if lemma_name is None:
            _ = list(executor.run_to_finish_lemma_return_exec())
            if executor.execution_complete:
                break
        else:
            logger.info(f"Discovered lemma: {lemma_name}")
            lemmas_to_prove.append(lemma_name)
            executor.run_to_finish_lemma()

    return lemmas_to_prove


def _log_lemma_names(lemmas: List[str], logger: logging.Logger) -> None:
    """Log lemma names, truncating if there are too many."""
    if len(lemmas) > 20:
        logger.info(f"Lemma names: {lemmas[:10]} ... {lemmas[-10:]}")
    else:
        logger.info(f"Lemma names: {lemmas}")


# Module-level wrapper for multiprocessing compatibility with Python 3.14t (forkserver)
def _get_all_lemmas_wrapper(
    ret_dict: Dict[str, Any],
    proof_exec_callback: ProofExecutorCallback,
    logger: logging.Logger,
    path: str
) -> None:
    """
    Wrapper function to get all lemmas - must be at module level for Python 3.14t pickling.

    Args:
        ret_dict: Shared dictionary for returning results
        proof_exec_callback: Callback for proof executor
        logger: Logger instance
        path: File path being processed
    """
    try:
        ret_dict["lemmas"] = get_all_lemmas(proof_exec_callback, logger)
    except Exception:
        logger.exception(f"Exception occurred while getting all lemmas in file: {path}")


def discover_lemmas_with_timeout(
    path: str,
    proof_exec_callback: ProofExecutorCallback,
    timeout: int,
    logger: logging.Logger
) -> Optional[List[str]]:
    """
    Discover lemmas in a file with a timeout using parallel execution.

    Args:
        path: Path to the proof file
        proof_exec_callback: Callback for proof executor
        timeout: Timeout in seconds
        logger: Logger instance

    Returns:
        List of lemma names if successful, None if timeout or error
    """
    logger.info(f"Getting all lemmas in file: {path} with timeout: {timeout} seconds")

    # Get the appropriate executor (free-threading for 3.14t+, multiprocessing for older)
    executor = get_executor()

    # Execute with timeout
    return_dict, _ = executor.execute_with_timeout(
        target=_get_all_lemmas_wrapper,
        args=(proof_exec_callback, logger, path),
        timeout=timeout
    )

    if return_dict is None or "lemmas" not in return_dict:
        logger.info(f"Failed to get all lemmas in file: {path}, moving on to the next file.")
        return None

    return list(return_dict["lemmas"])
