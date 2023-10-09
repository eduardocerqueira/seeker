#date: 2023-10-09T17:06:56Z
#url: https://api.github.com/gists/e19d6c03334957f0f72ae59c0583d647
#owner: https://api.github.com/users/ShangjinTang

#!/bin/bash

### stage 3 ####
# check if framework is actually using GPU
###

python3 -c "import tensorflow as tf; print('TensorFlow: Version: ' + tf.__version__ + ', GPU Available: ', bool(len(tf.config.list_physical_devices('GPU'))))"

python3 -c "import torch; print('PyTorch: Version: ' + torch.__version__ + ', GPU Available: ', torch.cuda.is_available())"