import os
import sys
import subprocess
import unittest
from unittest import TestCase


class TestSimpleCLI(TestCase):
    """Test the simple CLI for Lean 4 proof execution using subprocess."""

    @classmethod
    def setUpClass(cls):
        """Set up test environment."""
        # Set Lean version to match test project (optional, defaults to 4.24.0)
        # The test project uses 4.21.0 (see data/test/lean4_proj/lean-toolchain)
        os.environ["LEAN_VERSION"] = "4.21.0"

        # Define test project paths
        cls.test_project = "data/test/lean4_proj"
        cls.test_file = "Lean4Proj/Temp.lean"

        # CLI command - use python -m to invoke the module
        cls.cli_command = [sys.executable, "-m", "copra.simple"]

    def test_cli_basic_proof(self):
        """Test CLI with a basic theorem proof."""
        result = subprocess.run(
            self.cli_command + [
                "--project", self.test_project,
                "--file", self.test_file,
                "--theorem", "test",
                "--timeout", "200",
            ],
            capture_output=True,
            text=True,
            env=os.environ.copy()
        )

        print(f"\n{'='*60}")
        print(f"CLI Output:")
        print(result.stdout)
        if result.stderr:
            print(f"CLI Errors:")
            print(result.stderr)
        print(f"Exit code: {result.returncode}")
        print(f"{'='*60}")

        # Verify the command succeeded
        self.assertEqual(result.returncode, 0, f"CLI should succeed. stderr: {result.stderr}")
        self.assertIn("PROOF SUCCEEDED", result.stdout)

    def test_cli_with_output_file(self):
        """Test CLI with output file option."""
        import tempfile
        import os as os_module

        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            output_file = f.name

        try:
            result = subprocess.run(
                self.cli_command + [
                    "--project", self.test_project,
                    "--file", self.test_file,
                    "--theorem", "test",
                    "--output", output_file,
                ],
                capture_output=True,
                text=True,
                env=os.environ.copy()
            )

            print(f"\n{'='*60}")
            print(f"CLI Output:")
            print(result.stdout)
            print(f"Exit code: {result.returncode}")
            print(f"{'='*60}")

            # Verify the command succeeded
            self.assertEqual(result.returncode, 0, f"CLI should succeed. stderr: {result.stderr}")

            # Verify output file was created and contains proof
            self.assertTrue(os_module.path.exists(output_file), "Output file should exist")
            with open(output_file, 'r') as f:
                content = f.read()
                self.assertIn("Theorem: test", content)
                self.assertIn("SUCCESS", content)

            print(f"\nOutput file content:")
            print(content)

        finally:
            # Clean up output file
            if os_module.path.exists(output_file):
                os_module.remove(output_file)

    def test_cli_with_invalid_project(self):
        """Test CLI with non-existent project path."""
        result = subprocess.run(
            self.cli_command + [
                "--project", "non_existent_project",
                "--file", self.test_file,
                "--theorem", "test",
            ],
            capture_output=True,
            text=True,
            env=os.environ.copy()
        )

        print(f"\n{'='*60}")
        print(f"CLI Output:")
        print(result.stdout)
        print(f"CLI Errors:")
        print(result.stderr)
        print(f"Exit code: {result.returncode}")
        print(f"{'='*60}")

        # Verify the command failed
        self.assertNotEqual(result.returncode, 0, "CLI should fail with invalid project path")
        self.assertIn("does not exist", result.stderr)

    def test_cli_help(self):
        """Test CLI help output."""
        result = subprocess.run(
            self.cli_command + ["--help"],
            capture_output=True,
            text=True
        )

        print(f"\n{'='*60}")
        print(f"CLI Help Output:")
        print(result.stdout)
        print(f"{'='*60}")

        # Verify help output
        self.assertEqual(result.returncode, 0)
        self.assertIn("Simple CLI for running COPRA with Lean 4", result.stdout)
        self.assertIn("--project", result.stdout)
        self.assertIn("--file", result.stdout)
        self.assertIn("--theorem", result.stdout)
        self.assertIn("--timeout", result.stdout)
        self.assertIn("copra-lean-prover", result.stdout)

    def test_cli_custom_settings(self):
        """Test CLI with custom timeout and temperature."""
        result = subprocess.run(
            self.cli_command + [
                "--project", self.test_project,
                "--file", self.test_file,
                "--theorem", "test",
                "--timeout", "300",
                "--temperature", "0.8",
                "--proof-retries", "3",
            ],
            capture_output=True,
            text=True,
            env=os.environ.copy()
        )

        print(f"\n{'='*60}")
        print(f"CLI Output (custom settings):")
        print(result.stdout)
        print(f"Exit code: {result.returncode}")
        print(f"{'='*60}")

        # Verify command completed (may or may not succeed depending on theorem)
        self.assertIn("Execution time:", result.stdout)


class TestSimpleCLIIntegration(TestCase):
    """Integration tests for the simple CLI module imports."""

    def test_cli_module_imports(self):
        """Test that all required components can be imported."""
        from copra.simple import (
            SimpleLean4Config,
            SimpleLean4Runner,
            ProofCallback,
            ProofSearchResult
        )

        # Verify all imports succeeded
        self.assertIsNotNone(SimpleLean4Config)
        self.assertIsNotNone(SimpleLean4Runner)
        self.assertIsNotNone(ProofCallback)
        self.assertIsNotNone(ProofSearchResult)

    def test_config_to_experiments_conversion(self):
        """Test that SimpleLean4Config correctly converts to Experiments object."""
        from copra.simple import SimpleLean4Config

        config = SimpleLean4Config(
            project="data/test/lean4_proj",
            file_path="Lean4Proj/Temp.lean",
            theorem_name="test",
            timeout=200,
            temperature=0.7
        )

        # Convert to Experiments object
        experiments = config.to_experiments()

        # Verify conversion
        self.assertEqual(experiments.benchmark.name, "simple_lean4")
        self.assertEqual(experiments.eval_settings.timeout_in_secs, 200)
        self.assertEqual(experiments.eval_settings.temperature, 0.7)
        self.assertEqual(experiments.eval_settings.gpt_model_name, "gpt-5-mini")
        self.assertEqual(len(experiments.benchmark.datasets), 1)


if __name__ == "__main__":
    # Run with verbose output
    unittest.main(verbosity=2)
