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
LAMMPS_SRC=
LIBTORCH_DIR=

# 适用于BSCC-N16R4
cmake_prefix_path=${LIBTORCH_DIR}"/share/cmake;"${LIBTORCH_DIR}
echo $cmake_prefix_path
module load miniforge3/24.11 cudnn/9.6.0.74_cuda12 cuda/${CUDA_VERSION} gcc/11.4.0
source activate $CONDA_ENV_NAME

ssh -CfNg -L 7897:127.0.0.1:7897 ${CLIENT_NODE}
export https_proxy=http://127.0.0.1:7897
export http_proxy=http://127.0.0.1:7897
git config --global http.proxy http://127.0.0.1:7897
git config --global https.proxy http://127.0.0.1:7897
export PATH=$CONDA_PREFIX/nvvm/bin:$PATH
# 添加cicc到PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$CONDA_PREFIX/nvvm/lib64:$LD_LIBRARY_PATH
export LIBRARY_PATH=$CONDA_PREFIX/nvvm/lib64:$LIBRARY_PATH
export CPATH=$CONDA_PREFIX/nvvm/include:$CPATH

export CUDA_ROOT=$CONDA_PREFIX

cd $LAMMPS_SRC
mkdir -p build_$SLURM_JOB_ID
cd build_$SLURM_JOB_ID
which cicc
export CC=/usr/bin/x86_64-linux-gnu-gcc
export CXX=/usr/bin/x86_64-linux-gnu-g++
export FC=/data/apps/gcc/11.4.0/bin/gfortran
export OMPI_CC=$CC
export OMPI_CXX=$CXX
export OMPI_FC=$FC
export 
cmake \
    -D CMAKE_BUILD_TYPE=Debug \
    -D CMAKE_CXX_STANDARD=17 \
    -D CMAKE_CXX_COMPILER=$LAMMPS_SRC/lib/kokkos/bin/nvcc_wrapper \
    -D BUILD_MPI=ON \
    -D BUILD_OMP=ON \
    -D PKG_KOKKOS=ON \
    -D PKG_GPU=ON \
    -D CUDA_TOOLKIT_ROOT_DIR=$CONDA_PREFIX \
    -D FFT=KISS \
    -D GPU_API=cuda \
    -D CMAKE_CUDA_ARCHITECTURES=89 \
    -D GPU_ARCH=sm_89 \
    -D Kokkos_ENABLE_CUDA=ON \
    -D Kokkos_ARCH_AMDAVX=ON \
    -D MKL_INCLUDE_DIR="$CONDA_PREFIX/include" \
    -D CMAKE_PREFIX_PATH=$cmake_prefix_path \
    -D CUDA_CUDA_LIBRARY=True \
    -D CMAKE_LIBRARY_PATH=/data/apps/cuda/12.6/lib64/stubs \
    -D CMAKE_MPI_C_COMPILER=mpicc \
    -D CMAKE_MPI_CXX_COMPILER=mpicxx \
    -D PKG_ML-MACE=ON \
    -D CMAKE_INSTALL_PREFIX=$LAMMPS_SRC/install \
    $LAMMPS_SRC/cmake

make -j8 VERBOSE=1
if [ $? -ne 0 ]; then
    echo "CMake configuration failed."
    exit 1
fi
echo "CMake configuration succeeded."
echo "install path: $LAMMPS_SRC/install"
make install 
