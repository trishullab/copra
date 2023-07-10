# copra
COPRA: An in-COntext PRoof Agent which uses LLMs like GPTs to prove theorems in formal languages.

## Setup Steps:
1. Install OCaml first. Use the instructions here: https://opam.ocaml.org/doc/Install.html . The opam version used in this project is 2.1.3 (OCaml 4.14.0). Note that OCaml officially only supports Linux installations. One can use WSL on Windows machines.

2. Install Coq on Linux. The Coq version used in this project is 8.15.2. 
```
sudo apt install build-essential unzip bubblewrap
sh <(curl -sL https://raw.githubusercontent.com/ocaml/opam/master/shell/install.sh)
```

3. Create a `Miniconda` environment and activate it.

4. Change to the root working directory, and run the setup script i.e. `./src/scripts/setup.sh` fromt the root directory.