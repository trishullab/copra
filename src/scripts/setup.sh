if [[ ! -d "src/scripts" ]]; then
    # Raise an error if the scripts directory is not present
    echo "Please run this script from the root of the repository, cannot find src/scripts"
    exit 1
fi
# Assume that conda is activated
# Don't run without activating conda
# Check if Conda is activated
conda_status=$(conda info | grep "active environment" | cut -d ':' -f 2 | tr -d '[:space:]')
if [[ $conda_status == "None" ]] || [[ $conda_status == "base" ]]; then
    echo "Please activate conda before running this script"
    exit 1
fi
conda install pip
conda_bin=$(conda info | grep "active env location" | cut -d ':' -f 2 | tr -d '[:space:]')
pip_exe="$conda_bin/bin/pip"
ls -l $pip_exe
echo "Installing dependencies..."
echo "Installing Elan (Lean version manager) ..."
# # For Lean:
# # https://leanprover-community.github.io/install/debian_details.html
curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh
echo "Installed Elan (Lean version manager) successfully!"
echo "Installing Lean (lean:3.42.1) ..."
elan toolchain install leanprover-community/lean:3.42.1
elan override set leanprover-community/lean:3.42.1
echo "Installed Lean (lean:3.42.1) successfully!"
# # For installing leanproject
echo "Installing leanproject..."
$pip_exe install --user mathlibtools
echo "Installed leanproject successfully!"
echo "Installing OCaml (opam)..."
opam init -a --compiler=4.07.1
eval `opam config env`
opam update
# # For Coq:
echo "Installing Coq..."
opam pin add coq 8.10.2
opam pin -y add menhir 20190626
# # For SerAPI:
echo "Installing SerAPI (for interacting with Coq from Python)..."
opam install -y coq-serapi
echo "Installing Dpdgraph (for generating dependency graphs)..."
opam repo add coq-released https://coq.inria.fr/opam/released
opam install -y coq-dpdgraph
# Python dependencies
echo "Installing Python dependencies..."
$pip_exe install --user -r requirements.txt
(
    # Build CompCert
    echo "Building CompCert..."
    pushd ./data/benchmarks
    set -euv
    cd CompCert
    if [[ ! -f "Makefile.config" ]]; then
        ./configure x86_64-linux
    fi
    make -j `nproc`
    popd
    echo "CompCert built successfully!"
    # Ignore some proofs in CompCert
    # ./src/scripts/patch_compcert.sh
) || exit 1
echo "Building Simple Benchmark..."
pushd ./data/test/coq/custom_group_theory
cd theories
make
cd ..
popd
echo "Building Simple Benchmark done!"
echo "Setup complete!"