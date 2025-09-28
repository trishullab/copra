[![Build Status](https://github.com/trishullab/copra/actions/workflows/ci.yaml/badge.svg)](https://github.com/trishullab/copra/actions/workflows/ci.yaml)
[![PyPI version](https://img.shields.io/pypi/v/copra-theorem-prover.svg)](https://pypi.org/project/copra-theorem-prover/)
[![PyPI downloads](https://img.shields.io/pypi/dm/copra-theorem-prover.svg)](https://pypi.org/project/copra-theorem-prover/)
# copra
COPRA: An in-COntext PRoof Agent which uses LLMs like GPTs to prove theorems in formal languages.

# Setup Steps:
## Quick Setup for Lean 4:
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

## Full Setup for Coq and Lean:
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

## Setting up OpenAI API and Running Experiments:
1. You need to create a file `.secrets/openai_key.json` in the root directory of the project with the OpenAI API key. The file should contain the following:
```
{
    "organization": "<your-organization-id>",
    "api_key": "<your-api-key>"
}
```

2. The experiments are not necessarily thread safe. So, it is recommended to run them sequentially. The commands to run the desired experiments can be found in the file `./src/copra/main/config/experiments.yaml`.

3. Run the following command to run the experiments:
```bash
python src/copra/main/eval_benchmark.py
#^ This will run the experiments mentioned in the file `./src/copra/main/config/experiments.yaml`.
# Change the file path in the command above to run other experiments.
```

## Important Note:
The ITP projects must be built before running COPRA. Make sure that the switch is set correctly while running it for Coq projects because the Coq projects can be using different versions of Coq. 

## Paper:
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
