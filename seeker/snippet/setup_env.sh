#date: 2023-01-26T16:51:36Z
#url: https://api.github.com/gists/e2e4397e21e32654b24c5d6ffb1a6cb4
#owner: https://api.github.com/users/gotomypc

#!/usr/bin/env bash

set -e

root_dir="$(git rev-parse --show-toplevel)"
conda_dir="${root_dir}/conda"
env_dir="${root_dir}/env"

cd "${root_dir}"

# 1. Install conda at ./conda
if [ ! -d "${conda_dir}" ]; then
    printf "* Installing conda\n"
    curl --silent -L -o miniconda.sh "http://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh"
    bash ./miniconda.sh -b -f -p "${conda_dir}"
fi
eval "$("${conda_dir}/bin/conda" shell.bash hook)"

# 2. Create test environment at ./env
if [ ! -d "${env_dir}" ]; then
    printf "* Creating a test environment with PYTHON_VERSION=%s\n" "${PYTHON_VERSION}\n"
    conda create --prefix "${env_dir}" -y python="${PYTHON_VERSION}"
fi
conda activate "${env_dir}"

# 3. Install nightly PyTorch
conda install "pytorch==${PYTORCH_VERSION}" "cudatoolkit=${CU_VERSION}" -c pytorch