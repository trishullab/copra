[![Build Status](https://github.com/trishullab/copra/actions/workflows/ci.yaml/badge.svg)](https://github.com/trishullab/copra/actions/workflows/ci.yaml)
[![PyPI version](https://img.shields.io/pypi/v/copra-theorem-prover.svg)](https://pypi.org/project/copra-theorem-prover/)
[![PyPI downloads](https://img.shields.io/pypi/dm/copra-theorem-prover.svg)](https://pypi.org/project/copra-theorem-prover/)
# COPRA

COPRA: An in-COntext PRoof Agent which uses LLMs like GPTs to prove theorems in formal languages.

## Table of Contents
- [What's New](#whats-new)
- [Setup](#setup)
  - [Quick Setup for Lean 4](#quick-setup-for-lean-4)
  - [Python 3.14t Setup](#python-314t-setup-optional)
  - [Full Setup for Coq and Lean](#full-setup-for-coq-and-lean)
  - [vLLM Setup for Open Source Models](#vllm-setup-for-open-source-models)
- [Simple CLI for Lean 4](#simple-cli-for-lean-4)
- [Running Experiments](#running-experiments)
  - [API Setup](#api-setup)
  - [Running the miniF2F Benchmark](#running-the-minif2f-benchmark)
  - [Starting Required Services](#starting-required-services)
  - [Parallel Theorem Execution](#parallel-theorem-execution-new)
- [Latest Evaluation Results](#latest-evaluation-results-new)
- [Citation](#citation)

---

## What's New

### 🎯 Simple CLI for Lean 4 (NEW!)
Streamlined command-line interface for quick theorem proving - no complex YAML configuration needed! Prove theorems with a single command, use environment variables for API keys, and get real-time progress output.

[Jump to Simple CLI Guide →](#simple-cli-for-lean-4)

### 🤖 vLLM Support for Open Source Models (NEW!)
Run open-source LLMs locally (Llama, Mistral, DeepSeek, etc.) with GPU acceleration. OpenAI-compatible API, automatic server management, and support for any Hugging Face model compatible with vLLM.

[Jump to vLLM Setup →](#vllm-setup-for-open-source-models)

### 🚀 Parallel Theorem Execution (NEW!)
Execute proof search for multiple theorems in parallel to speed up evaluations on multi-core systems. Automatically uses threading for Python 3.14t+ and multiprocessing for older versions.

[Jump to Parallel Execution Setup →](#parallel-theorem-execution-new)

### 🐍 Python 3.14t Free-Threading Support (NEW!)
Full support for Python 3.14+ with free-threaded (GIL-free) execution. Automatically selects best parallel execution strategy based on Python version and GIL status.

### 📊 Latest Evaluation Results (NEW!)
Recent MiniF2F Test Lean 4 results with GPT-OSS-20b achieved **42.8% pass@5** success rate, including solving 1 IMO problem (`imo_1959_p1`) with low compute budget. [View logs →](docs/static/selected_results/gpt_oss_20b_evals_pass_at_5.log)

[Jump to Full Results →](#latest-evaluation-results-new)

---

## Setup

### Quick Setup for Lean 4
```bash
# 1. Install COPRA
pip install copra-theorem-prover

# 2. Set Lean version (default: 4.24.0) and install REPL
export LEAN_VERSION="4.21.0"  # Must match your project version
install-lean-repl

# 3. Build the interface (ensure $HOME/.elan/bin is in PATH)
source $HOME/.elan/env  # Optional: if lean --version fails
install-itp-interface
```

> **Note:** Tested on Linux. Windows users should use WSL.

### Python 3.14t Setup (Optional)
```bash
# Create and activate environment
conda create -n py314-ft python=3.14 python-freethreading -c conda-forge
conda activate py314-ft

# Install COPRA
pip install copra-theorem-prover  # or: pip install -e . (for development)

# Run experiments (automatically detects Python version)
python -m copra.main.run --config-name lean4_simple_experiment
```

> **Note:** Python 3.14t is experimental. COPRA automatically selects the best parallel execution strategy: free-threading (GIL disabled), threading (GIL enabled), or multiprocessing (Python < 3.14).

### Full Setup for Coq and Lean
```bash
# 1. Install Coq dependencies and opam (Linux only, use WSL on Windows)
sudo apt install build-essential unzip bubblewrap
sh <(curl -sL https://raw.githubusercontent.com/ocaml/opam/master/shell/install.sh)

# 2. Add to ~/.bashrc (adjust path if ~/.opam/default doesn't exist)
export PATH="/home/$USER/.opam/default/bin:$PATH"
export PATH="/home/$USER/.elan/bin:$PATH"

# 3. Setup Python environment and install Lean 4
# (Follow Quick Setup for Lean 4 steps above)
```

> **Note:** See [OCaml installation guide](https://opam.ocaml.org/doc/Install.html) for details.

### vLLM Setup for Open Source Models
```bash
# Standard vLLM models (Llama, Mistral, DeepSeek, etc.)
pip install copra-theorem-prover[os_models]

# GPT-OSS-20b (custom vLLM build with reasoning token support)
pip install copra-theorem-prover[gpt_oss]
pip install --pre torch --extra-index-url https://download.pytorch.org/whl/nightly/cu128
# Adjust CUDA version (cu118, cu121, cu128) based on your GPU
```

**Usage:**
```bash
# vLLM server starts automatically on port 48000 (override with VLLM_PORT)
python -m copra.main.eval_benchmark eval_settings=my_vllm_config benchmark=miniF2F
```

**Supported Models:** `vllm:openai/gpt-oss-20b`, `vllm:codellama/CodeLlama-7b-Instruct-hf`, `vllm:meta-llama/Llama-2-7b-chat-hf`, `vllm:EleutherAI/llemma_7b`, `vllm:deepseek-ai/deepseek-math-7b-instruct`, or any HuggingFace model compatible with vLLM.

> **Requirements:** Python ≤ 3.12 and CUDA-capable GPU. See [vLLM issue #26480](https://github.com/vllm-project/vllm/issues/26480) for known reasoning token limitations.

---

## Simple CLI for Lean 4

🎯 **NEW:** A streamlined command-line interface for quick theorem proving without complex configuration!

The simple CLI provides an easy way to prove theorems with minimal setup - perfect for quick experiments, testing, or integration into other tools.

**Quick Example:**
```bash
# Set your API key
export OPENAI_API_KEY="sk-..."

# Prove a theorem (using installed script)
copra-lean-prover \
  --project data/test/lean4_proj \
  --file Lean4Proj/Temp.lean \
  --theorem test

# Or use as a Python module
python -m copra.simple \
  --project data/test/lean4_proj \
  --file Lean4Proj/Temp.lean \
  --theorem test
```

**Key Features:**
- ✅ No complex YAML configuration needed
- ✅ Environment variable support for API keys (12-factor app pattern)
- ✅ Simple command-line arguments for common settings
- ✅ Real-time progress output
- ✅ Modular architecture ready for REST API integration

**Common Usage:**
```bash
# Prove all theorems in a file
copra-lean-prover --project <path> --file <file> --theorem "*"

# Override timeout and temperature
copra-lean-prover \
  --project <path> --file <file> --theorem <name> \
  --timeout 300 --temperature 0.8

# Save proof to file
copra-lean-prover \
  --project <path> --file <file> --theorem <name> \
  --output proof.txt
```

**Available Options:**
- `--timeout` - Timeout in seconds (default: 200)
- `--temperature` - LLM sampling temperature (default: 0.7)
- `--proof-retries` - Number of retry attempts (default: 4)
- `--main-prompt` - Custom system prompt
- `--conv-prompt` - Custom conversation prompt
- `--model` - LLM model to use (default: gpt-5-mini)
- `--output` - Save proof to file
- `--verbose` - Detailed logging

📖 **[Full Documentation →](src/copra/simple/README.md)**

---

## Running Experiments

### API Setup
**Option 1: Environment Variables (Recommended)**
```bash
# OpenAI
export OPENAI_API_KEY="sk-..."

# Anthropic
export ANTHROPIC_API_KEY="sk-ant-..."

# AWS Bedrock
export AWS_ACCESS_KEY_ID="..."
export AWS_SECRET_ACCESS_KEY="..."
export AWS_REGION="us-east-1"
```

**Option 2: Secrets File**
Create `.secrets/openai_key.json`:
```json
{"organization": "<your-org-id>", "api_key": "<your-api-key>"}
```

### Running Experiments
```bash
# Auto-detects Python version (recommended)
python src/copra/main/run.py --config-name lean4_simple_experiment
```

> **Note:** See `./src/copra/main/config/experiments.yaml` for available configurations.

### Running the miniF2F Benchmark
```bash
# Setup Lean 4.21.0 (required for miniF2F)
export LEAN_VERSION="4.21.0"
install-lean-repl && install-itp-interface

# Run with GPT-OSS-20b
python -m copra.main.run --config-name miniF2F_lean4_easy_to_hard

# Run with OpenAI models (requires API key)
python -m copra.main.eval_benchmark \
  benchmark=miniF2F_lean4_test_easy_to_hard \
  eval_settings=n_60_dfs_gpt4_o_no_retrieve_no_ex
```

**Results:** `.log/proofs/eval_driver/dfs/miniF2F_lean4_test_easy_to_hard/<timestamp>/`

### Starting Required Services

- **Lean 4:** No setup needed (COPRA manages REPL automatically)
- **Isabelle:** COPRA auto-manages PISA service (default port: 17000, override with `PISA_PORT`)
- **Coq:** Ensure correct version is active (`coqc --version`) and project is built (`make`)
- **vLLM/Llama:** COPRA auto-starts services (logs in `.log/evals/benchmark/<name>/<timestamp>/`)

> **Important:** ITP projects must be built before running COPRA. For Coq, ensure the correct opam switch is active.

### Parallel Theorem Execution (NEW!)

**Quick Start:**
```bash
# Enable with auto-detected workers (CPU cores / 2)
export ENABLE_PARALLEL_THEOREMS=True
python src/copra/main/run.py --config-name lean4_simple_experiment

# Custom worker count
export MAX_PARALLEL_WORKERS=4
```

**How it works:** Automatically uses threading (Python 3.14t+) or multiprocessing (Python < 3.14). Disabled by default.

**Configuration:**
- `ENABLE_PARALLEL_THEOREMS`: Enable/disable (default: `False`)
- `MAX_PARALLEL_WORKERS`: Worker count (default: auto-detected)

**Tips:** Match workers to CPU cores, ensure services handle concurrent requests, use sequential mode for <4 theorems.

**Troubleshooting:** Reduce workers if you encounter memory/service errors.

---

## Latest Evaluation Results (NEW!)

You can find selected recent evaluation results here:
- MiniF2F Test Lean 4 (v4.21.0) - GPT-OSS-20b (low reasoning tokens) (no retrieval):
  - About 42.798% overall success rate on pass@5
  - Solves 1 IMO problem (`imo_1959_p1`)
  - Decent performance for low compute budget
  - See logs: [execution_logs](docs/static/selected_results/gpt_oss_20b_evals_pass_at_5.log)



## Important Notes

- The ITP projects must be built before running COPRA
- For Coq projects, ensure the correct switch/version is active
- Services (PISA, Llama) are automatically managed by COPRA
- Check logs in `.log/evals/benchmark/<name>/<timestamp>/` for debugging

---

## Citation
You can cite our paper:
```
@inproceedings{thakur2024context,
  title={An in-context learning agent for formal theorem-proving},
  author={Thakur, Amitayush and Tsoukalas, George and Wen, Yeming and Xin, Jimmy and Chaudhuri, Swarat},
  booktitle={First Conference on Language Modeling},
  year={2024}
}
```
Our paper can be found here: [OpenReview](https://openreview.net/forum?id=V7HRrxXUhN#discussion) and [ArXiv](https://arxiv.org/abs/2310.04353)
