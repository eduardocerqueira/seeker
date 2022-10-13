#date: 2022-10-13T17:17:52Z
#url: https://api.github.com/gists/82f6c5bd3d2141cc6e3faf17f06baefc
#owner: https://api.github.com/users/yippiez

from PIL import Image
import numpy as np
import pdf2image
import sys
import os

pdf_path = sys.argv[1] # stdin path input
new_folder_path = pdf_path[:-4]
image_array = pdf2image.convert_from_path(pdf_path)

try:
    os.mkdir(new_folder_path)
except FileExistsError:
    print(f"{new_folder_path}: Folder already exists moving on")

d = len(str(len(image_array))) + 1

for i in range(len(image_array)):
    Image.fromarray(np.asarray(image_array[i])).save(os.path.join(new_folder_path, os.path.basename(new_folder_path) + f"_{i+1:0{d}}.jpg"))