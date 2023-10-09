#date: 2023-10-09T17:06:56Z
#url: https://api.github.com/gists/e19d6c03334957f0f72ae59c0583d647
#owner: https://api.github.com/users/ShangjinTang

#!/bin/bash

### stage 2 ####
# verify the nvidia driver + cuda + cudnn installation
# install TensorFlow and PyTorch
###

# verify the installation
nvidia-smi
nvcc -V

# install TensorFlow GPU
python3 -m pip install nvidia-cudnn-cu11==8.6.0.163 tensorflow==2.13.0

# install PyTorch GPU
python3 -m pip install torch==2.1.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118