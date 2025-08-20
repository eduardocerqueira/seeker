#date: 2025-08-20T17:07:57Z
#url: https://api.github.com/gists/ab4cf4e2b1ee15a2d86ad764ffe808aa
#owner: https://api.github.com/users/bio-punk

#!/bin/bash
#SBATCH --job-name=TestJob
#SBATCH --gpus=1
#SBATCH --time=23:59:59
#SBATCH --output=test_%j_output.log
#SBATCH --error=test_%j_output.log
#SBATCH

CONDA_ENV_NAME=
CUDA_VERSION=
LAMMPS_SRC=

module load miniforge3/24.11 cudnn/9.6.0.74_cuda12 cuda/${CUDA_VERSION}
source activate ${CONDA_ENV_NAME}
alias mpirun='mpirun --mca opal_cuda_support 1'
# module load ucx
# alias mpirun='mpirun --mca pml ucx'
export PATH=$LAMMPS_SRC/install:$PATH
mpirun -np 1 lmp -h
echo ----------------------------------------------------------------

python mace/mace/cli/create_lammps_model.py my_mace.model
if [ $? -ne 0 ]; then
    echo "Failed to create LAMMPS model."
    exit 1
fi

mpirun -np 1 lmp -k on g 1 -sf kk -in test_in.lammps
