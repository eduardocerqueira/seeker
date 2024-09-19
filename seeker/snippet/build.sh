#date: 2024-09-19T16:38:44Z
#url: https://api.github.com/gists/e53448fcad4df388798f905fb02fa218
#owner: https://api.github.com/users/cpcloud

#!/usr/bin/env bash

set -euo pipefail

export TZ=America/New_York DEBIAN_FRONTEND=noninteractive

apt-get update -y -qq
apt-get install -y build-essential software-properties-common git -qq
apt-get update -y -qq
apt-get autoremove python3 -y -qq

apt-get install -y -qq python3.12 xz-utils python3.12-venv ninja-build jq neovim curl

pushd / || exit
curl -LsSO 'https://www.apache.org/dyn/closer.lua/arrow/arrow-17.0.0/apache-arrow-17.0.0.tar.gz?action=download'

tar xzf apache-arrow-17.0.0.tar.gz

mv apache-arrow-17.0.0 arrow
popd || exit

git clone https://github.com/emscripten-core/emsdk.git /emsdk

python3.12 -m venv /pyodide-pyarrow
source /pyodide-pyarrow/bin/activate

pushd /emsdk || exit
./emsdk install 3.1.58
./emsdk activate 3.1.58
source /emsdk/emsdk_env.sh
popd || exit

pip install pyodide-build==0.26.2

pushd /arrow/cpp || exit

jq \
  '(.configurePresets[] | select(.name == "features-emscripten").cacheVariables) += {ARROW_FILESYSTEM: "ON", ARROW_ORC: "OFF", ARROW_SUBSTRAIT: "OFF", ARROW_ACERO: "OFF"}' \
  < CMakePresets.json > /tmp/out.json
mv /tmp/out.json CMakePresets.json

emcmake cmake --preset "ninja-release-emscripten"
ninja install

pushd ../python || exit
pyodide build
popd || exit

popd || exit