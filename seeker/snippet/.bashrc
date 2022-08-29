#date: 2022-08-29T17:14:40Z
#url: https://api.github.com/gists/a6c7c4603c5811fdcafce19b1e22d01f
#owner: https://api.github.com/users/maawad

# CUDA INCLUDES
#export CUDA_HOME=/usr/local/cuda
export ROCM_HOME=/opt/rocm/
export CUDA_HOME=/usr/local/cuda-11.5
export PATH=${CUDA_HOME}/bin:$PATH
export PATH=${ROCM_HOME}/bin:$PATH
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${CUDA_HOME}/lib:$LD_LIBRARY_PATH

# CONDA INCLUDES
export CONDA_PKGS_DIRS="/home/conda/packages"
export CONDA_ENVS_DIRS="/home/$USER/.conda/envs":"/opt/miniconda3/envs"

# TBB
export TBB_INSTALL_DIR=$HOME/dev/oneTBB
export TBB_INCLUDE=$TBB_INSTALL_DIR/include
export TBB_LIBRARY_RELEASE=$TBB_INSTALL_DIR/release/gnu_9.3_cxx11_64_relwithdebinfo

export PATH=${TBB_INCLUDE}:$PATH
export LD_LIBRARY_PATH=${TBB_LIBRARY_RELEASE}:$LD_LIBRARY_PATH


#CMake 3.19
# export CMAKE_DIR="/home/mawad/cmake/cmake-3.19.5/bin"
#export CMAKE_DIR="/opt/miniconda3/bin"
# export PATH=${CMAKE_DIR}:$PATH

#alias
alias  watch-gpu="watch -n 0.5 nvidia-smi"
alias sgqueue="squeue -O jobarrayid:15,partition:11,qos:8,numcpus:6,tres-per-job:14,username:12,state:9,timeused:10,reason:10,name:40 -S -t,-Q,-M"
alias sginfo="sinfo -o \"%20N  %10c  %10m  %25f  %40G \""
alias mps-gpu="nvidia-cuda-mps-control -d"
alias mps-gpu-exit="echo quit | nvidia-cuda-mps-control"
alias watch='watch '

function pretty_csv {
    column -t -s, -n "$@" | less -F -S -X -K
}

export LLVM_PATH=/opt/rocm/llvm
export PATH=${LLVM_PATH}/bin:$PATH