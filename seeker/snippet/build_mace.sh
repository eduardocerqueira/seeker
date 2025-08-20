#date: 2025-08-20T17:07:57Z
#url: https://api.github.com/gists/ab4cf4e2b1ee15a2d86ad764ffe808aa
#owner: https://api.github.com/users/bio-punk

#!/bin/bash
#SBATCH --job-name=lammps_pna
#SBATCH --gpus=1
#SBATCH

#手动设置以下变量
CLIENT_NODE=
CONDA_ENV_NAME=
CUDA_VERSION=

module load miniforge3/24.11 cudnn/9.6.0.74_cuda12 cuda/${CUDA_VERSION} gcc/11.4.0
source activate $CONDA_ENV_NAME

pip install ./mace