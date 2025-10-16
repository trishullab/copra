[![Build Status](https://github.com/trishullab/copra/actions/workflows/ci.yaml/badge.svg)](https://github.com/trishullab/copra/actions/workflows/ci.yaml)
[![PyPI version](https://img.shields.io/pypi/v/copra-theorem-prover.svg)](https://pypi.org/project/copra-theorem-prover/)
[![PyPI downloads](https://img.shields.io/pypi/dm/copra-theorem-prover.svg)](https://pypi.org/project/copra-theorem-prover/)
# COPRA

COPRA: An in-COntext PRoof Agent which uses LLMs like GPTs to prove theorems in formal languages.

## Table of Contents
- [What's New](#whats-new)
- [Setup](#setup)
  - [Quick Setup for Lean 4](#quick-setup-for-lean-4)
  - [Python 3.14t Setup (Free-threaded Python)](#python-314t-setup-free-threaded-python---optional)
  - [Full Setup for Coq and Lean](#full-setup-for-coq-and-lean)
- [Running Experiments](#running-experiments)
  - [Setting up OpenAI API](#setting-up-openai-api-and-running-experiments)
  - [Starting Required Services](#starting-required-services)
  - [Parallel Theorem Execution](#parallel-theorem-execution-new)
- [Citation](#paper)

---

## What's New

### 🚀 Parallel Theorem Execution (NEW!)
COPRA now supports executing proof search for **multiple theorems in parallel**! This significantly speeds up evaluation on multi-core systems.

**Features:**
- ✅ **Automatic Python Version Detection**: Uses threading for Python 3.14t+ and multiprocessing for older versions
- ✅ **Configurable Worker Count**: Control parallelism with environment variables
- ✅ **Backward Compatible**: Disabled by default - opt-in via environment variable
- ✅ **Better Resource Utilization**: Leverage all CPU cores for faster evaluations

[Jump to Parallel Execution Setup →](#parallel-theorem-execution-new)

### 🐍 Python 3.14t Free-Threading Support
COPRA now fully supports Python 3.14+ with free-threaded (GIL-free) execution for improved performance!

**Automatic Execution Mode Selection:**
- **Python 3.14t+ with GIL disabled:** Native free-threading for true parallel execution
- **Python 3.14t+ with GIL enabled:** Threading (still benefits from improved performance)
- **Python < 3.14:** Multiprocessing (traditional approach)

---

## Setup

### Quick Setup for Lean 4
1. Install itp-interface using the following command: (Our package is available on PyPI: https://pypi.org/project/copra-theorem-prover/)
```bash
pip install copra-theorem-prover
```

2. Run the following command to prepare the REPL for Lean 4. (The default version is 4.7.0-rc2. You can change the version by setting the `LEAN_VERSION` environment variable. If no version is set, then 4.7.0-rc2 is used.)
>NOTE: The Lean 4 version must match the version of the Lean 4 project you are working with.
```bash
export LEAN_VERSION="4.15.0"
install-lean-repl
```

3. Run the following command to build the REPL for Lean 4. Make sure that `lean --version` returns the correct version before running the command below. If not then check if `$HOME/.elan/bin` is in your path. Recommended to run `source $HOME/.elan/env` before running the command below.
```bash
install-itp-interface
```

>NOTE: These steps are only tested on Linux. For Windows, you can use WSL. These steps will not setup the Coq interface.

### Python 3.14t Setup (Free-threaded Python - Optional)
🆕 **NEW:** COPRA now supports Python 3.14+ with free-threaded (GIL-free) support for improved performance!

1. **Create a Conda environment with Python 3.14t (free-threaded):**
```bash
# Create environment with free-threaded Python 3.14
conda create -n py314-ft python=3.14 python-freethreading -c conda-forge

# Activate the environment
conda activate py314-ft

# Verify Python version and free-threading support
python --version  # Should show Python 3.14.x
```

2. **Install COPRA theorem prover:**
```bash
# Install from PyPI
pip install copra-theorem-prover

# OR for development, install from source
pip install -e .
```

3. **Run experiments with Python 3.14t:**
```bash
# Use run.py which automatically detects Python 3.14+ and uses Hydra-free mode
python -m copra.main.run --config-name lean4_simple_experiment

# Or if installed from source
python src/copra/main/run.py --config-name lean4_simple_experiment
```

> **Note:** Python 3.14t is experimental. Some packages may show compatibility warnings (especially Pydantic and OpenAI), but COPRA has been refactored to work with Python 3.14t.

**🚀 Automatic Parallel Execution Selection:**
COPRA automatically detects your Python version and chooses the best parallel execution strategy:
- **Python 3.14t+ with GIL disabled:** Uses native free-threading for true parallel execution
- **Python 3.14t+ with GIL enabled:** Uses threading (still benefits from improved performance)
- **Python < 3.14:** Uses multiprocessing (traditional approach)

Check your execution mode:
```bash
python demo_parallel_execution.py
```

### Full Setup for Coq and Lean
1. Install OCaml first. Use the instructions here: https://opam.ocaml.org/doc/Install.html. Note that OCaml officially only supports Linux installations. One can use WSL on Windows machines.

2. Run the following to install Coq on Linux.
```
sudo apt install build-essential unzip bubblewrap
sh <(curl -sL https://raw.githubusercontent.com/ocaml/opam/master/shell/install.sh)
```

3. Add the following to your `.bashrc` file: (sometimes the path `~/.opam/default` might not exist, so use the directory with version number present in the `~/.opam` directory)
```
export PATH="/home/$USER/.opam/default/bin:$PATH"
```

4. Create a `Miniconda` environment and activate it.

5. Run the commands for installing the Lean 4 interface as mentioned in [Quick Setup for Lean 4](#quick-setup-for-lean-4).

6. Add the following to your `.bashrc` file for Lean:
```
export PATH="/home/$USER/.elan/bin:$PATH"
```

---

## Running Experiments

### Setting up OpenAI API and Running Experiments
1. You need to create a file `.secrets/openai_key.json` in the root directory of the project with the OpenAI API key. The file should contain the following:
```
{
    "organization": "<your-organization-id>",
    "api_key": "<your-api-key>"
}
```

2. The experiments are not necessarily thread safe. So, it is recommended to run them sequentially. The commands to run the desired experiments can be found in the file `./src/copra/main/config/experiments.yaml`.

3. Run the following command to run the experiments:

**For Python 3.14+ (with free-threaded support):**
```bash
python src/copra/main/run.py --config-name lean4_simple_experiment
# This uses a Hydra-free implementation compatible with Python 3.14+
# You can change the config name to run different experiments
```

**For Python < 3.14:**
```bash
python src/copra/main/eval_benchmark.py
# This will run the experiments mentioned in the file `./src/copra/main/config/experiments.yaml`.
# Change the file path in the command above to run other experiments.
```

**Universal command (auto-detects Python version):**
```bash
python src/copra/main/run.py --config-name lean4_simple_experiment
# This automatically uses Hydra-free mode for Python 3.14+ and Hydra mode for older versions
```

> **Note:** `run.py` is the recommended entry point for all Python versions. It automatically detects your Python version and uses the appropriate implementation (Hydra-free for 3.14+, standard Hydra for older versions).

### Starting Required Services

Before running COPRA, you need to ensure the required services are running based on your proof language:

#### For Lean 4 Projects:
No additional services are required. Lean 4 uses a built-in REPL that COPRA manages automatically.

#### For Isabelle Projects:
You need to start the PISA (Portal for Isabelle) service:

1. **Set the PISA port** (optional, defaults to 17000):
```bash
export PISA_PORT=17000
```

2. **Start the PISA service:**
```bash
# COPRA will automatically start the PISA service when needed
# You can also manually start it if required by your setup
```

3. **Check if PISA is running:**
```bash
# COPRA will log whether PISA service is up or needs to be restarted
```

> **Note:** COPRA automatically manages the PISA service lifecycle, including health checks and restarts. You'll see log messages about PISA service status during execution.

#### For Coq Projects:
Ensure the correct Coq version is active:

1. **Check your Coq version:**
```bash
coqc --version
```

2. **Switch Coq version if needed** (using opam):
```bash
opam switch <your-coq-version>
eval $(opam env)
```

3. **Build your Coq project** before running COPRA:
```bash
cd /path/to/your/coq/project
make  # or your project's build command
```

#### For LLM Services (Llama models):
If using non-OpenAI models (like Llama):

1. COPRA will automatically initialize the Llama service when needed
2. Check the logs in `.log/evals/benchmark/<benchmark-name>/<timestamp>/llama.log`
3. If the service goes down, COPRA will automatically restart it

> **Important:** The ITP projects must be built before running COPRA. Make sure the correct version/switch is active for Coq projects, as different projects may use different Coq versions.

### Parallel Theorem Execution (NEW!)

COPRA now supports executing proof search for **multiple theorems in parallel**, significantly speeding up evaluations on multi-core systems.

#### Quick Start

**Enable parallel execution:**
```bash
export ENABLE_PARALLEL_THEOREMS=True
python src/copra/main/run.py --config-name lean4_simple_experiment
```

**Control number of workers:**
```bash
export ENABLE_PARALLEL_THEOREMS=True
export MAX_PARALLEL_WORKERS=4  # Use 4 parallel workers
python src/copra/main/run.py --config-name lean4_simple_experiment
```

#### Configuration Options

| Environment Variable | Description | Default | Example |
|---------------------|-------------|---------|---------|
| `ENABLE_PARALLEL_THEOREMS` | Enable/disable parallel theorem execution | `False` | `True` or `False` |
| `MAX_PARALLEL_WORKERS` | Maximum number of parallel workers | Auto (CPU cores / 2) | `4`, `8`, `16` |

#### How It Works

**Sequential Execution (Default):**
- Theorems are processed one at a time
- Each theorem proof runs in isolation with timeout
- Original behavior is preserved

**Parallel Execution (Enabled):**
- Multiple theorems are processed concurrently
- Each theorem proof still runs in its own process/thread with timeout
- Results are collected as they complete
- **Automatic Python version detection:**
  - **Python < 3.14:** Uses `ProcessPoolExecutor` (multiprocessing)
  - **Python 3.14t+:** Uses `ThreadPoolExecutor` (threading with free-threading support)

#### Examples

**Example 1: Basic parallel execution**
```bash
# Enable parallel execution with auto-detected worker count
export ENABLE_PARALLEL_THEOREMS=True
python -m copra.main.run --config-name lean4_simple_experiment
```

**Example 2: Custom worker count**
```bash
# Use 8 parallel workers for large theorem sets
export ENABLE_PARALLEL_THEOREMS=True
export MAX_PARALLEL_WORKERS=8
python -m copra.main.run --config-name lean4_simple_experiment
```

**Example 3: Python 3.14t with free-threading**
```bash
# On Python 3.14t, this will automatically use threading
conda activate py314-ft
export ENABLE_PARALLEL_THEOREMS=True
python -m copra.main.run --config-name lean4_simple_experiment
```

**Example 4: Conservative parallel execution**
```bash
# Use only 2 workers for resource-constrained systems
export ENABLE_PARALLEL_THEOREMS=True
export MAX_PARALLEL_WORKERS=2
python -m copra.main.run --config-name lean4_simple_experiment
```

#### Performance Tips

1. **Match workers to CPU cores:** Set `MAX_PARALLEL_WORKERS` to your number of physical CPU cores
2. **Consider theorem complexity:** Complex theorems benefit more from parallelism
3. **Monitor resources:** Watch CPU, memory, and I/O usage during execution
4. **Service capacity:** Ensure LLM services can handle concurrent requests
5. **Sequential for small sets:** Use sequential execution for fewer than 4 theorems (less overhead)

#### Limitations

- **Memory usage:** Parallel execution uses more memory (multiple processes/threads running simultaneously)
- **Shared resources:** Services (Llama, Isabelle) must handle concurrent requests
- **Time budget tracking:** Time budgets are tracked per-theorem, not globally across parallel executions

#### Troubleshooting

**Issue: Out of memory errors**
- **Solution:** Reduce `MAX_PARALLEL_WORKERS` to `2` or `4`

**Issue: Service connection errors**
- **Solution:** Reduce parallel workers or use sequential execution for service-heavy workloads

**Issue: Slower than sequential execution**
- **Cause:** Small theorem sets, I/O-bound workloads, or limited CPU cores
- **Solution:** Disable parallel execution or reduce worker count

---

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
