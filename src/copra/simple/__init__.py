"""
Simple interface for running COPRA proofs with Lean 4.

This module provides a simplified API for proving theorems without
the complexity of full benchmark evaluation. It's designed to be
used both from CLI and as a library for REST APIs.
"""

from copra.simple.core import (
    SimpleLean4Config,
    SimpleLean4Runner,
    ProofCallback,
)
from itp_interface.rl.proof_tree import ProofSearchResult

__all__ = [
    "SimpleLean4Config",
    "SimpleLean4Runner",
    "ProofCallback",
    "ProofSearchResult",  # Re-export from itp_interface
]
