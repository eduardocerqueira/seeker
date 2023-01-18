#date: 2023-01-18T16:48:21Z
#url: https://api.github.com/gists/068364d8d178a32afec712a5ff8a8474
#owner: https://api.github.com/users/maslyankov

#!/bin/bash
lspci -nn | grep 089a
ls /dev/apex_0

cd ~
mkdir coral
cd coral/
wget https://github.com/hjonnala/snippets/raw/main/wheels/python3.10/pycoral-2.0.0-cp310-cp310-linux_x86_64.whl
wget https://github.com/hjonnala/snippets/raw/main/wheels/python3.10/tflite_runtime-2.5.0.post1-cp310-cp310-linux_x86_64.whl

# Due to errors with command "sudo apt-get install python3-pycoral"...
sudo apt install python3-pip
pip install tflite_runtime-2.5.0.post1-cp310-cp310-linux_x86_64.whl
pip install pycoral-2.0.0-cp310-cp310-linux_x86_64.whl

git clone https://github.com/google-coral/pycoral.git
cd pycoral
bash examples/install_requirements.sh classify_image.py
python3 examples/classify_image.py --model test_data/mobilenet_v2_1.0_224_inat_bird_quant_edgetpu.tflite --labels test_data/inat_bird_labels.txt --input test_data/parrot.jpg