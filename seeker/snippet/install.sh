#date: 2023-01-26T16:51:36Z
#url: https://api.github.com/gists/e2e4397e21e32654b24c5d6ffb1a6cb4
#owner: https://api.github.com/users/gotomypc

#!/usr/bin/env bash

set -e

root_dir="$(git rev-parse --show-toplevel)"
conda_dir="${root_dir}/conda"
env_dir="${root_dir}/env"

cd "${root_dir}"

# 0. Activate conda env
eval "$("${conda_dir}/bin/conda" shell.bash hook)"
conda activate "${env_dir}"

# 1. Install build tools
conda install pkg-config
pip install cmake ninja

# 2. torchaudio
python setup.py install