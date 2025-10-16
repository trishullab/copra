#!/usr/bin/env python3

"""
Checkpoint Manager Module

This module handles checkpoint and proof result management,
including saving, loading, and querying checkpoint state.
"""

import logging
from typing import Optional

from copra.main.config import EvalRunCheckpointInfo, EvalProofResults
from itp_interface.rl.proof_tree import ProofSearchResult


class CheckpointManager:
    """Manages checkpoints and proof results."""

    def __init__(
        self,
        checkpoint_info: EvalRunCheckpointInfo,
        proof_results: EvalProofResults,
        logger: logging.Logger
    ):
        """
        Initialize the checkpoint manager.

        Args:
            checkpoint_info: Checkpoint information
            proof_results: Proof results
            logger: Logger instance
        """
        self.checkpoint_info = checkpoint_info
        self.proof_results = proof_results
        self.logger = logger

    def should_skip_file(self, path: str, skip_files_in_checkpoint: bool) -> bool:
        """
        Determine if a file should be skipped based on checkpoint state.

        Args:
            path: File path
            skip_files_in_checkpoint: Whether to skip files that are in checkpoint

        Returns:
            True if file should be skipped, False otherwise
        """
        if not skip_files_in_checkpoint:
            return False

        if path not in self.checkpoint_info.theorem_maps:
            return False

        self.logger.info(f"Skipping the file: {path} as it was already attempted before.")

        # Log existing results for this file
        if path in self.proof_results.theorem_map:
            for lemma_name, proof_res_chkpt in self.proof_results.theorem_map[path].items():
                self.logger.info(f"Dumping proof search result:\n{proof_res_chkpt}")
                self.logger.info(f"Prover for lemma: {lemma_name} in file {path} completed.")

        return True

    def add_path_to_maps(self, path: str) -> None:
        """
        Add a path to both checkpoint and proof result maps.

        Args:
            path: File path to add
        """
        self.checkpoint_info.add_path_to_maps(path)
        self.proof_results.add_path_to_maps(path)

    def get_proof_checkpoint(self, path: str, lemma_name: str) -> Optional[ProofSearchResult]:
        """
        Get existing proof checkpoint for a lemma.

        Args:
            path: File path
            lemma_name: Lemma name

        Returns:
            ProofSearchResult if exists, None otherwise
        """
        return self.proof_results.theorem_map.get(path, {}).get(lemma_name, None)

    def should_attempt_proof(
        self,
        path: str,
        lemma_name: str,
        attempt_idx: int,
        max_retry_attempts: int
    ) -> bool:
        """
        Determine if a proof should be attempted based on checkpoint state.

        Args:
            path: File path
            lemma_name: Lemma name
            attempt_idx: Current attempt index
            max_retry_attempts: Maximum retry attempts allowed

        Returns:
            True if proof should be attempted, False otherwise
        """
        proof_res_chkpt = self.get_proof_checkpoint(path, lemma_name)

        if proof_res_chkpt is None:
            return True

        if not proof_res_chkpt.proof_found and \
           proof_res_chkpt.additional_info["attempt_idx"] < max_retry_attempts - 1:
            return True

        # Log that we're skipping
        proof_res_attempt_idx = proof_res_chkpt.additional_info["attempt_idx"]
        if proof_res_attempt_idx == attempt_idx:
            self.logger.info(f"Dumping proof search result:\n{proof_res_chkpt}")
            self.logger.info(f"Prover for lemma: {lemma_name} in file {path} completed.")
        else:
            self.logger.info(
                f"Skipping the attempt for proving lemma: {lemma_name} in file {path} "
                f"as it was already attempted before."
            )

        return False

    def save_proof_result(
        self,
        path: str,
        lemma_name: str,
        proof_result: ProofSearchResult,
        success: bool
    ) -> None:
        """
        Save a proof result and update checkpoint.

        Args:
            path: File path
            lemma_name: Lemma name
            proof_result: Proof search result
            success: Whether the proof was successful
        """
        self.proof_results.add_theorem_to_maps(path, lemma_name, proof_result)
        self.checkpoint_info.add_theorem_to_maps(path, lemma_name, success)

        if success:
            self.logger.info(f"Prover for lemma: {lemma_name} in file {path} completed.")
        else:
            self.logger.info(f"Failed to prove lemma: {lemma_name} in file {path}")

    def get_previous_queries(self, path: str, lemma_name: str) -> int:
        """
        Get the number of previous queries for a lemma.

        Args:
            path: File path
            lemma_name: Lemma name

        Returns:
            Number of previous queries, or 0 if no checkpoint exists
        """
        proof_res_chkpt = self.get_proof_checkpoint(path, lemma_name)
        if proof_res_chkpt is not None and "queries" in proof_res_chkpt.additional_info:
            return proof_res_chkpt.additional_info["queries"]
        return 0

    def get_previous_attempt_idx(self, path: str, lemma_name: str, current_attempt_idx: int) -> int:
        """
        Get the previous attempt index for a lemma.

        Args:
            path: File path
            lemma_name: Lemma name
            current_attempt_idx: Current attempt index

        Returns:
            Previous attempt index + 1, or current_attempt_idx if no checkpoint exists
        """
        proof_res_chkpt = self.get_proof_checkpoint(path, lemma_name)
        if proof_res_chkpt is not None and "attempt_idx" in proof_res_chkpt.additional_info:
            return proof_res_chkpt.additional_info["attempt_idx"] + 1
        return current_attempt_idx
