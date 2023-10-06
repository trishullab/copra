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

5. Add the following to your `.bashrc` file:
```
export PATH="/home/<username>/.opam/default/bin:$PATH"
```

6. You need to create a file `.secrets/openai_key.json` in the root directory with the OpenAI API key. The file should contain the following:
```
{
    "organization": "<your-organization-id>",
    "api_key": "<your-api-key>"
}
```

7. The experiments are not necessarily thread safe. So, it is recommended to run them sequentially. The commands to run the desired experiments can be found in the file `./src/main/config/experiments.yaml`.