#!/usr/bin/env python3
"""
Simple CLI for running COPRA with Lean 4.

This provides a streamlined command-line interface for proving theorems
without the complexity of full benchmark evaluation.
"""

import argparse
import logging
import sys
import os
from pathlib import Path
from typing import Optional

# Add project root to path
root_dir = f"{__file__.split('copra')[0]}"
if root_dir not in sys.path:
    sys.path.append(root_dir)

from copra.simple.core import (
    SimpleLean4Config,
    SimpleLean4Runner,
    ProofCallback
)
from itp_interface.rl.proof_tree import ProofSearchResult
from itp_interface.tools.log_utils import setup_logger


class ConsoleProofCallback(ProofCallback):
    """Callback that prints proof progress to console."""

    def __init__(self, verbose: bool = True, theorem_name: Optional[str] = None):
        """
        Initialize console callback.

        Args:
            verbose: If True, print detailed step information
            theorem_name: Optional theorem name for display
        """
        self.verbose = verbose
        self.theorem_name = theorem_name

    def on_start(self, theorem_name: str) -> None:
        """Print when proof starts."""
        self.theorem_name = theorem_name
        print(f"\n{'='*60}")
        print(f"Starting proof for: {theorem_name}")
        print(f"{'='*60}\n")

    def on_complete(self, result: ProofSearchResult, execution_time: float) -> None:
        """Print final result."""
        print(f"\n{'='*60}")
        if result.proof_found:
            print(f"✓ PROOF SUCCEEDED")
        else:
            print(f"✗ PROOF FAILED")
        print(f"{'='*60}")
        print(f"Execution time: {execution_time:.2f}s")
        print(f"\n{str(result)}")
        print()


def save_proof_to_file(result: ProofSearchResult, execution_time: float, theorem_name: str, output_file: str) -> None:
    """
    Save proof result to a file.

    Args:
        result: ProofSearchResult from the proof search
        execution_time: Time taken for proof execution
        theorem_name: Name of the theorem
        output_file: Path to output file
    """
    with open(output_file, 'w') as f:
        f.write(f"Theorem: {theorem_name}\n")
        f.write(f"Status: {'SUCCESS' if result.proof_found else 'FAILED'}\n")
        f.write(f"Execution time: {execution_time:.2f}s\n")
        f.write("=" * 60 + "\n\n")

        # Write as JSON (ProofSearchResult supports JSON serialization)
        try:
            import json
            f.write("JSON:\n")
            # Use to_dict() if available, otherwise convert to dict manually
            if hasattr(result, 'to_dict'):
                json_data = result.to_dict()
            else:
                # Fallback: use dataclass fields
                from dataclasses import asdict
                json_data = asdict(result)
            f.write(json.dumps(json_data, indent=2, default=str))
            f.write("\n\n")
        except Exception as e:
            f.write(f"JSON serialization failed: {e}\n\n")

        # Also write string representation
        f.write("String representation:\n")
        f.write(str(result))
        f.write("\n")

    print(f"Proof saved to: {output_file}")


def create_parser() -> argparse.ArgumentParser:
    """Create and configure argument parser."""
    parser = argparse.ArgumentParser(
        description='Simple CLI for running COPRA with Lean 4',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Prove a specific theorem
  copra-lean-prover \\
    --project data/test/lean4_proj \\
    --file Lean4Proj/Temp.lean \\
    --theorem test

  # Prove all theorems in a file
  copra-lean-prover \\
    --project data/test/lean4_proj \\
    --file Lean4Proj/Temp.lean \\
    --theorem "*"

  # Override settings and save to file
  copra-lean-prover \\
    --project data/test/lean4_proj \\
    --file Lean4Proj/Temp.lean \\
    --theorem test \\
    --timeout 300 \\
    --temperature 0.8 \\
    --proof-retries 3 \\
    --output proof.txt

  # Use custom prompts
  copra-lean-prover \\
    --project data/test/lean4_proj \\
    --file Lean4Proj/Temp.lean \\
    --theorem test \\
    --main-prompt data/prompts/system/my-prompt.md \\
    --conv-prompt data/prompts/conversation/my-conv.md

Environment Variables:
  OPENAI_API_KEY         OpenAI API key (or use .secrets/openai_key.json)
  ANTHROPIC_API_KEY      Anthropic API key (or use .secrets/anthropic_key.json)
  AWS_ACCESS_KEY_ID      AWS access key for Bedrock
  AWS_SECRET_ACCESS_KEY  AWS secret key for Bedrock
  AWS_REGION             AWS region for Bedrock
        """
    )

    # Required arguments
    parser.add_argument(
        '--project',
        type=str,
        required=True,
        help='Path to the Lean 4 project directory'
    )
    parser.add_argument(
        '--file',
        type=str,
        required=True,
        help='Path to the Lean file (relative to project)'
    )
    parser.add_argument(
        '--theorem',
        type=str,
        required=True,
        help='Theorem name to prove, or "*" for all theorems in the file'
    )

    # Optional settings
    parser.add_argument(
        '--output',
        '-o',
        type=str,
        help='Save proof to file'
    )
    parser.add_argument(
        '--timeout',
        '-t',
        type=int,
        default=200,
        help='Timeout in seconds for proof attempts (default: 200)'
    )
    parser.add_argument(
        '--temperature',
        '-T',
        type=float,
        default=0.7,
        help='Temperature for LLM sampling (default: 0.7)'
    )
    parser.add_argument(
        '--proof-retries',
        '-r',
        type=int,
        default=4,
        help='Number of retries for proof attempts (default: 4)'
    )

    # Prompt settings
    parser.add_argument(
        '--main-prompt',
        type=str,
        default='data/prompts/system/simplified-lean4-proof-agent-with-dfs.md',
        help='Path to main system prompt (default: simplified-lean4-proof-agent-with-dfs.md)'
    )
    parser.add_argument(
        '--conv-prompt',
        type=str,
        default='data/prompts/conversation/simplified-lean4-proof-agent-dfs-multiple.md',
        help='Path to conversation prompt (default: simplified-lean4-proof-agent-dfs-multiple.md)'
    )
    parser.add_argument(
        '--uses-simplified-prompt',
        type=lambda x: x.lower() == 'true',
        default=True,
        help='Use simplified prompt format (default: true)'
    )

    # Model settings
    parser.add_argument(
        '--model',
        type=str,
        default='gpt-5-mini',
        help='LLM model to use (default: gpt-5-mini)'
    )
    parser.add_argument(
        '--max-steps',
        type=int,
        default=60,
        help='Maximum proof steps per episode (default: 60)'
    )

    # Logging
    parser.add_argument(
        '--verbose',
        '-v',
        action='store_true',
        help='Print detailed step information'
    )
    parser.add_argument(
        '--quiet',
        '-q',
        action='store_true',
        help='Suppress all output except final result'
    )
    parser.add_argument(
        '--log-file',
        type=str,
        help='Save logs to file'
    )

    return parser


def main():
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()

    # Make sure that the project path exists
    if not os.path.exists(args.project):
        print(f"Error: Project path does not exist: {args.project}", file=sys.stderr)
        sys.exit(1)
    
    # Check if the file path is relative to the project or absolute
    # Check if the file path is relative
    file_path = Path(args.file)
    project_path = Path(args.project)

    # Check if file_path is relative to project_path
    if not file_path.is_absolute():
        if file_path.is_relative_to(project_path):
            full_file_path = str(file_path)
        else:
            # Make it relative to project path
            full_file_path = str(project_path / file_path)
    else:
        full_file_path = str(file_path)
    
    assert os.path.exists(full_file_path), f"Error: File path does not exist: {full_file_path}"
    args.file = full_file_path

    # Create config from arguments
    config = SimpleLean4Config(
        project=args.project,
        file_path=args.file,
        theorem_name=args.theorem,
        timeout=args.timeout,
        temperature=args.temperature,
        proof_retries=args.proof_retries,
        main_prompt=args.main_prompt,
        conv_prompt=args.conv_prompt,
        uses_simplified_prompt=args.uses_simplified_prompt,
        model_name=args.model,
        max_steps_per_episode=args.max_steps
    )

    # Setup logging
    if args.log_file:
        logger = setup_logger(
            "SimpleLean4CLI",
            args.log_file,
            logging.DEBUG if args.verbose else logging.INFO,
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    else:
        logger = logging.getLogger("SimpleLean4CLI")
        logger.setLevel(logging.DEBUG if args.verbose else logging.INFO)
        if not args.quiet:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s'
            ))
            logger.addHandler(handler)

    # Create runner
    runner = SimpleLean4Runner(config, logger)

    # Create callback
    callback = None if args.quiet else ConsoleProofCallback(verbose=args.verbose)

    # Run proof
    try:
        result, execution_time = runner.run_proof(callback=callback)

        # Save to file if requested
        if args.output:
            save_proof_to_file(result, execution_time, config.theorem_name, args.output)

        # Exit with appropriate code
        sys.exit(0 if result.proof_found else 1)

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.exception(f"Fatal error: {e}")
        print(f"\nFatal error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
