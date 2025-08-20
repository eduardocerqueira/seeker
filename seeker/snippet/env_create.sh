#date: 2025-08-20T17:07:57Z
#url: https://api.github.com/gists/ab4cf4e2b1ee15a2d86ad764ffe808aa
#owner: https://api.github.com/users/bio-punk

#!/bin/bash

# 本脚本在登录节点运行
# 以下变量需要手动配置
CONDA_ENV_NAME=
CUDA_VERSION="12.4"
LAMMPS_SRC=
# END

# 以下内容适配BSCC-N16R4

module load miniforge3/24.11 cudnn/9.6.0.74_cuda12 cuda/${CUDA_VERSION}

conda create -n ${CONDA_ENV_NAME} \
    -c nvidia/label/cuda-${CUDA_VERSION}.0 \
    -c conda-forge \
    python=3.10 \
    "numpy<2" \
    mpi4py \
    openmpi \
    -y

source activate ${CONDA_ENV_NAME}

conda install \
    -c nvidia/label/cuda-${CUDA_VERSION}.0/linux-64 \
    -c conda-forge \
    -c pytorch \
    cuda-nvrtc \
    cuda-cudart \
    cuda-version=${CUDA_VERSION} \
    cuda-nvcc \
    cuda-cudart-dev \
    cuda-nvtx \
    libcusparse-dev \
    libcublas-dev \
    libcusolver-dev \
    libcurand-dev \
    cuda-nvvm-tools \
    "numpy<2" \
    -y

conda install \
    -c nvidia/label/cuda-${CUDA_VERSION}.0/linux-64 \
    -c conda-forge \
    -c pytorch \
    cuda-cccl \
    cuda-cccl_linux-64 \
    cuda-crt-dev_linux-64 \
    cuda-crt-tools \
    cuda-cudart-dev_linux-64 \
    cuda-cudart-static_linux-64 \
    cuda-cudart_linux-64 \
    cuda-driver-dev_linux-64 \
    cuda-nvcc \
    cuda-nvcc-dev_linux-64 \
    cuda-nvcc-impl \
    cuda-nvcc-tools \
    cuda-nvcc_linux-64 \
    cuda-nvvm-dev_linux-64 \
    cuda-nvvm-impl \
    cuda-nvvm-tools \
    cuda-opencl \
    cuda-version=${CUDA_VERSION} \
    -y

pip install "numpy<2" torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124
python -c "import torch; print(torch._C._GLIBCXX_USE_CXX11_ABI)"
# 如果输出为True，则表示使用了C++11 ABI，否则为False

# 输出为False就需要安装libtorch
wget https://download.pytorch.org/libtorch/cu124/libtorch-shared-with-deps-2.4.0%2Bcu124.zip
cd $LAMMPS_SRC && cd .. && unzip libtorch-shared-with-deps-2.4.0+cu124.zip

# 输出为True，torch使用C++11 ABI时，将以下路径添加到-D CMAKE_PREFIX_PATH中
# python -c 'import torch;print(torch.utils.cmake_prefix_path)'

git config --global http.proxy http://127.0.0.1:7897
git config --global https.proxy http://127.0.0.1:7897

git clone --branch=mace --depth=1 https://github.com/ACEsuit/lammps ${LAMMPS_SRC}

cp nvcc_wrapper ${LAMMPS_SRC}/lib/kokkos/bin/nvcc_wrapper
# default_arch="sm_89"
# host_compiler='/usr/bin/x86_64-linux-gnu-g++'

# cp ./cuda.cmake ${CONDA_PREFIX}/lib/python3.10/site-packages/torch/share/cmake/Caffe2/public/cuda.cmake
# # 修改172行寻找nvtx3
# # set(USE_SYSTEM_NVTX ON)
# # 修改175行寻找nvtx3
# # find_path(nvtx3_dir NAMES nvtx3 PATHS "/data/apps/cuda/12.6/include")

conda install "cmake<=3.28.2" -c conda-forge -y
conda install -c conda-forge gdb -y
