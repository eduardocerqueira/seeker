#date: 2023-03-22T16:57:02Z
#url: https://api.github.com/gists/713ca4865eedb74ccdc31d01f1727bf8
#owner: https://api.github.com/users/vincentAgnus

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as get_module_version

import subprocess
import os

python_libs ="""
torch
tensorflow
tensorboard
numpy
pandas
scikit-image
scikit-learn
seaborn
simpleitk
tensorboard
torchvision
yaml
imgaug
yacs
yaml
pydicom
tqdm
sacred
cv2
"""

python_libs = python_libs.split()
#print(python_libs)

versions = dict()

for module in python_libs:
    try:
        versions[module] = get_module_version(module)
    except PackageNotFoundError:
        versions[module] = 'NOT FOUND'



#cuda version
try:
    import torch
    assert(torch.cuda.is_available())
    versions["cuda"] = torch.version.cuda
except:
    #code for second chance with tf ?
    try:
        versions["cuda"] = "NOTFOUND"
        #out = subprocess.check_output(["nvcc","--version"],shell=True, env=None)
        #proc = subprocess.Popen("nvcc --version",stdout=subprocess.PIPE,shell=True)
        #(out, err) = proc.communicate()
        #print(out)
    except Exception as e:
        #
        print(e)
        versions["cuda"] = "NOTFOUND"


try:
    import torch
    assert(torch.backends.cudnn.is_available())
    versions["cudnn"] = torch.backends.cudnn.version()
except:
    versions["cudnn"] = "NOTFOUND"




for module, version in versions.items():
    print(f"{module:<16}{version}")
