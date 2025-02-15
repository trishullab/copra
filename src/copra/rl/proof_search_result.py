#!/usr/bin/env python3

import sys
root_dir = f"{__file__.split('src')[0]}"
if root_dir not in sys.path:
    sys.path.append(root_dir)
import typing
from copra.tools.training_data_format import TrainingDataFormat
from dataclasses_json import dataclass_json
from dataclasses import dataclass

@dataclass_json
@dataclass
class ProofSearchResult(object):
    proof_file: typing.Optional[str]
    proof_found: bool
    lemma_name: str
    proof_steps: typing.List[TrainingDataFormat]
    proof_time_in_secs: float
    inferences_taken: int
    possible_failed_paths: int
    num_of_backtracks: int
    is_timeout: bool
    is_inference_exhausted: bool
    longest_success_path: int

    def __str__(self) -> str:
        try:
            lines = [line for step in self.proof_steps for line in step.proof_steps]
            proof_metadata = f"""
ProofFile: {self.proof_file}
LemmaName: {self.lemma_name}
SearchResult: {'[FAILED]' if not self.proof_found else '[SUCCESS]'}
IsInferenceExhausted: {self.is_inference_exhausted}
IsTimeout: {self.is_timeout}
LongestSuccessPath: {self.longest_success_path} 
StepsUsed: {self.inferences_taken}
SearchTimeInSecs: {self.proof_time_in_secs}
NumberOfBacktracks: {self.num_of_backtracks}
PossibleFailedPaths: {self.possible_failed_paths}
--------------------
"""
            all_proof_steps = "\n    ".join(lines[:-1]) if len(lines) > 1 else ""
            last_line = (lines[-1] if lines[-1] == "Qed." else f"    {lines[-1]}\n") if len(lines) > 0 else ""
            return f"""{self.lemma_name}
Proof.
    {all_proof_steps}
{last_line}
{proof_metadata}
"""
        except Exception:
            return f"------- UNABLE_TO_PRINT_INCOMPLETE_PROOF ----------"