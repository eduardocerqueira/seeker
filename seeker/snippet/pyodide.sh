#date: 2025-05-08T16:46:03Z
#url: https://api.github.com/gists/b5ca3d833f91db07f56750987b13c479
#owner: https://api.github.com/users/kazuki0824

apt update
apt install git build-essential curl nano -y
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

git clone https://github.com/kazuki0824/uv-cibuildwheel-ruff-coverage-template
cd uv-cibuildwheel-ruff-coverage-template

uv venv --python 3
. ./.venv/bin/activate

git clone https://github.com/emscripten-core/emsdk.git
cd emsdk
./emsdk install 4.0.6
./emsdk activate 4.0.6
source ./emsdk_env.sh
cd ..

curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- \
  -y \
  --default-toolchain nightly \
  --target wasm32-unknown-emscripten
. "$HOME/.cargo/env"


uvx --from pyodide-build --with pyodide-cli pyodide build