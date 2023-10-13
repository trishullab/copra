if [[ ! -d "src/scripts" ]]; then
    # Raise an error if the scripts directory is not present
    echo "Please run this script from the root of the repository, cannot find src/scripts"
    exit 1
fi
# Don't run without activating conda
# Check if Conda is activated
conda_status=$(conda info | grep "active environment" | cut -d ':' -f 2 | tr -d '[:space:]')
if [[ $conda_status == "None" ]] || [[ $conda_status == "base" ]]; then
    echo "Please activate conda environment before running this script"
    exit 1
fi
echo "Setting up Copra ..."
echo "[NOTE] The installation needs manual intervention on some steps. Please choose the appropriate option when prompted."
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
source $HOME/.elan/env
echo "Installing Lean (lean:3.42.1) ..."
elan toolchain install leanprover-community/lean:3.42.1
elan override set leanprover-community/lean:3.42.1
echo "Installed Lean (lean:3.42.1) successfully!"
export PATH=$PATH:$HOME/.elan/bin
# # # For installing leanproject
# echo "Installing leanproject..."
# $pip_exe install --user mathlibtools
# echo "Installed leanproject successfully!"
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
echo "Clone all git submodules..."
git submodule update --init --recursive
echo "Cloned all git submodules successfully!"
echo "Building Coq projects..."
(
    # Build CompCert
    echo "Building CompCert..."
    echo "This may take a while... (don't underestimate the time taken to build CompCert, meanwhile you can take a coffee break!)"
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
echo "Building Coq's Simple Benchmark..."
pushd ./data/test/coq/custom_group_theory
cd theories
make
cd ..
popd
echo "Building Coq's Simple Benchmark done!"
echo "Coq's Setup complete!"
echo "Building Lean's projects ..."
(
    # Build Lean's projects
    echo "Building miniF2F..."
    echo "This may take a while... (don't underestimate the time taken to build miniF2F, meanwhile you can take a coffee break!)"
    pushd ./data/benchmarks
    set -euv
    cd miniF2F
    leanpkg configure
    leanproject get-mathlib-cache # This allows us to use .olean files from mathlib without building them again
    leanproject build
    popd
    echo "miniF2F built successfully!"
) || exit 1
echo "Building Lean's Simple Benchmark..."
pushd ./data/test/lean_proj
leanproject build
popd
echo "Building Lean's Simple Benchmark done!"
echo "Building Lean's projects done!"
echo "Lean's Setup complete!"
echo "Copra Setup complete!"
