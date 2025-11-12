# Simple COPRA CLI for Lean 4

A streamlined command-line interface for running COPRA proofs with Lean 4, designed for ease of use without the complexity of full benchmark evaluation.

## Features

- **Simple Command-Line Interface**: Prove theorems with a single command
- **Environment Variable Support**: No need for `.secrets/` files - use standard environment variables
- **Real-Time Progress**: See proof steps as they execute
- **Flexible Configuration**: Override only the settings you need
- **Modular Architecture**: Core logic separated for future REST API support

## Installation

Install the COPRA package from PyPI:

```bash
pip install copra-theorem-prover
```

This will install the `copra-lean-prover` command globally.

For development, install from source:

```bash
git clone https://github.com/trishullab/copra.git
cd copra
pip install -e .
```

## Quick Start

### 1. Set Lean Version (Recommended)

It's recommended to set the Lean version to match your project. If not set, it defaults to `4.24.0`:

```bash
# Check your project's Lean version
cat data/test/lean4_proj/lean-toolchain
# Output: leanprover/lean4:v4.21.0

# Set the LEAN_VERSION environment variable to match
export LEAN_VERSION="4.21.0"
```

> **Note:** While optional (defaults to 4.24.0), setting `LEAN_VERSION` to match your project's `lean-toolchain` file is strongly recommended. Version mismatches may cause unexpected behavior or proof failures.

### 2. Set API Key

```bash
# For OpenAI models
export OPENAI_API_KEY="your-api-key"

# Or create a secrets file
mkdir -p .secrets
echo '{"api_key": "your-api-key"}' > .secrets/openai_key.json
```

### 3. Run a Proof

```bash
# After installation, use the command-line script:
copra-lean-prover \
  --project data/test/lean4_proj \
  --file Lean4Proj/Temp.lean \
  --theorem test

# Or use as a Python module:
python -m copra.simple \
  --project data/test/lean4_proj \
  --file Lean4Proj/Temp.lean \
  --theorem test
```

## Usage

After installing the package, you can use the `copra-lean-prover` command:

### Basic Usage

```bash
copra-lean-prover \
  --project <path-to-lean4-project> \
  --file <relative-path-to-lean-file> \
  --theorem <theorem-name>
```

### Prove All Theorems in a File

```bash
copra-lean-prover \
  --project data/test/lean4_proj \
  --file Lean4Proj/Temp.lean \
  --theorem "*"
```

### Override Settings

```bash
copra-lean-prover \
  --project data/test/lean4_proj \
  --file Lean4Proj/Temp.lean \
  --theorem test \
  --timeout 300 \
  --temperature 0.8 \
  --proof-retries 3 \
  --model gpt-5-mini
```

### Save Proof to File

```bash
copra-lean-prover \
  --project data/test/lean4_proj \
  --file Lean4Proj/Temp.lean \
  --theorem test \
  --output proof.txt
```

### Use Custom Prompts

```bash
copra-lean-prover \
  --project data/test/lean4_proj \
  --file Lean4Proj/Temp.lean \
  --theorem test \
  --main-prompt data/prompts/system/my-prompt.md \
  --conv-prompt data/prompts/conversation/my-conv.md
```

### Enable Verbose Logging

```bash
copra-lean-prover \
  --project data/test/lean4_proj \
  --file Lean4Proj/Temp.lean \
  --theorem test \
  --verbose \
  --log-file proof.log
```

> **Note:** You can also use `python -m copra.simple` instead of `copra-lean-prover` if you prefer.

## Command-Line Options

### Required Arguments

- `--project` - Path to the Lean 4 project directory
- `--file` - Path to the Lean file (relative to project)
- `--theorem` - Theorem name to prove, or "*" for all theorems

### Optional Settings

- `--output`, `-o` - Save proof to file
- `--timeout`, `-t` - Timeout in seconds (default: 200)
- `--temperature`, `-T` - LLM sampling temperature (default: 0.7)
- `--proof-retries`, `-r` - Number of retry attempts (default: 4)

### Prompt Settings

- `--main-prompt` - Path to main system prompt (default: simplified-lean4-proof-agent-with-dfs.md)
- `--conv-prompt` - Path to conversation prompt (default: simplified-lean4-proof-agent-dfs-multiple.md)
- `--uses-simplified-prompt` - Use simplified prompt format (default: true)

### Model Settings

- `--model` - LLM model to use (default: gpt-5-mini)
- `--max-steps` - Maximum proof steps per episode (default: 60)

### Logging

- `--verbose`, `-v` - Print detailed step information
- `--quiet`, `-q` - Suppress all output except final result
- `--log-file` - Save logs to file

## Environment Variables

The CLI uses environment variables for configuration. Here are all supported variables:

### Lean Version (Recommended)
```bash
# RECOMMENDED: Set Lean version to match your project
export LEAN_VERSION="4.21.0"

# Check your project's lean-toolchain file to find the correct version:
# cat your-project/lean-toolchain
# Output: leanprover/lean4:vX.XX.X
```

> **Note:** If not set, defaults to `4.24.0`. It's strongly recommended to set this to match your project's `lean-toolchain` file to avoid version mismatch issues.

### API Keys

The CLI checks for credentials in environment variables before falling back to `.secrets/` files:

#### OpenAI
```bash
export OPENAI_API_KEY="sk-..."
```

#### Anthropic
```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

#### AWS Bedrock
```bash
export AWS_ACCESS_KEY_ID="AKIA..."
export AWS_SECRET_ACCESS_KEY="..."
export AWS_REGION="us-east-1"
```

#### vLLM (Local Models)
```bash
export VLLM_API_KEY="EMPTY"
export VLLM_BASE_URL="http://127.0.0.1:48000/v1"
```

## Architecture

The simple CLI is organized into three layers:

```
src/copra/simple/
├── __init__.py          # Public API exports
├── core.py              # Core business logic (I/O agnostic)
├── cli.py               # Command-line interface
├── api.py               # (Future) REST API interface
└── __main__.py          # Module entry point
```

### Core Module (`core.py`)

Contains all business logic with no I/O dependencies:

- `SimpleLean4Config` - Configuration dataclass
- `SimpleLean4Runner` - Proof execution engine
- `ProofResult` - Structured result object
- `ProofCallback` - Callback interface for progress updates

This design allows the same logic to be reused for both CLI and REST API.

### CLI Module (`cli.py`)

Terminal-specific functionality:

- `ConsoleProofCallback` - Real-time console output
- `save_proof_to_file()` - File output
- Argument parsing and validation

## Examples

### Example 1: Basic Proof

```bash
# Set Lean version first
export LEAN_VERSION="4.21.0"
export OPENAI_API_KEY="sk-..."

# Run proof
copra-lean-prover \
  --project data/test/lean4_proj \
  --file Lean4Proj/Temp.lean \
  --theorem test
```

Output:
```
============================================================
Starting proof for: test
============================================================

2025-11-12 10:30:15 - INFO - Starting proof for theorem: test
2025-11-12 10:30:20 - INFO - Proof completed in 5.23s

============================================================
✓ PROOF SUCCEEDED
============================================================
Execution time: 5.23s

ProofSearchResult(
  proof_found=True,
  lemma_name='test',
  proof_time_in_secs=5.23,
  inferences_taken=8,
  proof_steps=[...],
  ...
)
```

### Example 2: Failed Proof

```bash
copra-lean-prover \
  --project data/test/lean4_proj \
  --file Lean4Proj/Temp.lean \
  --theorem difficult_theorem \
  --timeout 60
```

Output:
```
============================================================
Starting proof for: difficult_theorem
============================================================

2025-11-12 10:35:15 - INFO - Starting proof for theorem: difficult_theorem
2025-11-12 10:36:15 - WARNING - Timeout reached

============================================================
✗ PROOF FAILED
============================================================
Execution time: 60.00s

ProofSearchResult(
  proof_found=False,
  lemma_name='difficult_theorem',
  is_timeout=True,
  ...
)
```

### Example 3: Save to File

```bash
copra-lean-prover \
  --project data/test/lean4_proj \
  --file Lean4Proj/Temp.lean \
  --theorem test \
  --output proof.txt
```

Creates `proof.txt`:
```
Theorem: test
Status: SUCCESS
Execution time: 5.23s
============================================================

JSON:
{
  "proof_found": true,
  "lemma_name": "test",
  "proof_time_in_secs": 5.23,
  "inferences_taken": 8,
  "proof_steps": [...],
  ...
}

String representation:
ProofSearchResult(proof_found=True, lemma_name='test', ...)
```

## Programmatic Usage

You can also use the simple CLI as a library:

```python
from copra.simple import SimpleLean4Config, SimpleLean4Runner, ProofCallback, ProofSearchResult

# Create config
config = SimpleLean4Config(
    project="data/test/lean4_proj",
    file_path="Lean4Proj/Temp.lean",
    theorem_name="test",
    timeout=200,
    temperature=0.7
)

# Create custom callback (optional)
class MyCallback(ProofCallback):
    def on_start(self, theorem_name: str):
        print(f"Starting proof for: {theorem_name}")

    def on_complete(self, result: ProofSearchResult, execution_time: float):
        print(f"Completed in {execution_time:.2f}s")
        print(f"Proof {'succeeded' if result.proof_found else 'failed'}")

# Run proof
runner = SimpleLean4Runner(config)
result, execution_time = runner.run_proof(callback=MyCallback())

# Check result - result is a ProofSearchResult object
if result.proof_found:
    print(f"Proof: {result.proof_file}")
    print(f"Time: {result.proof_time_in_secs}s")
    print(f"Inferences: {result.inferences_taken}")

# Serialize to string (includes all info: proof, time, queries, steps, etc.)
print(str(result))
```

## Future: REST API

The modular architecture is designed to support a REST API in the future:

```python
# api.py (future)
from fastapi import FastAPI
from copra.simple.core import SimpleLean4Config, SimpleLean4Runner

app = FastAPI()

@app.post("/api/v1/prove")
async def prove_theorem(config: SimpleLean4Config):
    runner = SimpleLean4Runner(config)
    result, execution_time = runner.run_proof()

    # Return structured response
    return {
        "success": result.proof_found,
        "execution_time": execution_time,
        "lemma_name": result.lemma_name,
        "proof": result.proof_file if result.proof_found else None,
        "proof_time_in_secs": result.proof_time_in_secs,
        "inferences_taken": result.inferences_taken,
        "is_timeout": result.is_timeout,
        "details": str(result)  # Full string representation
    }
```

This will enable a web-based frontend in `src/app/` that calls the REST API.

## Differences from Full COPRA

The simple CLI removes complexity while maintaining core functionality:

| Feature | Full COPRA | Simple CLI |
|---------|------------|------------|
| Configuration | Multiple YAML files | Single command |
| API Keys | `.secrets/` files only | Environment variables + files |
| Benchmarks | Complex dataset configs | Direct project/file/theorem |
| Checkpointing | Full checkpoint support | No checkpointing |
| Parallel execution | Multiple theorems | Single theorem |
| Output | Complex log directories | Console + optional file |

## Troubleshooting

### Lean Version Mismatch

If you encounter unexpected errors or proof failures, check if your Lean version matches:

```bash
# Check your project's Lean version
cat your-project/lean-toolchain  # Shows: leanprover/lean4:v4.21.0

# Set LEAN_VERSION to match (defaults to 4.24.0 if not set)
export LEAN_VERSION="4.21.0"
```

> **Tip:** Version mismatches don't always cause immediate errors but may lead to subtle proof failures or unexpected behavior.

### API Key Not Found

```
Error: OpenAI API key not found. Set OPENAI_API_KEY environment variable...
```

Solution: Set the appropriate environment variable or create a secrets file.

### Theorem Not Found

```
Error: No lemmas discovered in Lean4Proj/Temp.lean
```

Solution: Check that the file path is correct and the theorem exists.

### Timeout

```
Proof FAILED - Reason: Timeout
```

Solution: Increase timeout with `--timeout 300` or simplify the theorem.

## Contributing

When adding features to the simple CLI, maintain the separation of concerns:

- Core logic goes in `core.py` (I/O agnostic)
- Terminal-specific code goes in `cli.py`
- Future API code will go in `api.py`

This ensures the same logic can be used across different interfaces.
