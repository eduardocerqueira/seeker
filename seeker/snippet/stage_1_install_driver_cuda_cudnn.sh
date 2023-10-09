#date: 2023-10-09T17:06:56Z
#url: https://api.github.com/gists/e19d6c03334957f0f72ae59c0583d647
#owner: https://api.github.com/users/ShangjinTang

#!/bin/bash

### stage 1 ####
# verify the system has a cuda-capable gpu
# download and install the nvidia cuda toolkit and cudnn
# setup environmental variables
###

### to verify your gpu is cuda enable check
lspci | grep -i nvidia

### remove previous installation
sudo apt purge '.*nvidia.*' '.*cuda.*' '.*cudnn.*'
sudo apt remove '.*nvidia.*' '.*cuda.*' '.*cudnn.*'
sudo rm /etc/apt/sources.list.d/cuda*
sudo apt-get autoremove && sudo apt-get autoclean
sudo rm -rf /usr/local/cuda*

### do system upgrade
sudo apt update && sudo apt upgrade -y
sudo apt install -y g++ freeglut3-dev build-essential libx11-dev libxmu-dev libxi-dev libglu1-mesa libglu1-mesa-dev

# install nvidia driver
sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt update
sudo apt install -y libnvidia-common-535 libnvidia-gl-535 nvidia-driver-535

# install cuda deb(network)
# Reference: https://developer.nvidia.com/cuda-11-8-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=deb_network
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update
sudo apt install -y cuda-11-8

# Note: you need to add below lines to ~/.bashrc or ~/.zshrc
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/."$(basename $SHELL)"rc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/."$(basename $SHELL)"rc

# install cuDNN v8.7
# Reference: https://developer.nvidia.com/cudnn
CUDNN_TAR_FILE="cudnn-linux-x86_64-8.7.0.84_cuda11-archive.tar.xz"
sudo wget https://developer.download.nvidia.com/compute/redist/cudnn/v8.7.0/local_installers/11.8/cudnn-linux-x86_64-8.7.0.84_cuda11-archive.tar.xz
sudo tar -xvf cudnn-linux-*.tar.xz
sudo mv cudnn-linux-x86_64-8.7.0.84_cuda11-archive cuda

# copy the following files into the cuda toolkit directory.
sudo cp -P cuda/include/cudnn.h /usr/local/cuda/include
sudo cp -P cuda/lib/libcudnn* /usr/local/cuda/lib64/
sudo chmod a+r /usr/local/cuda/lib64/libcudnn*

# reboot to solve "Failed to initialize NVML: Driver/library version mismatch"
sudo reboot
