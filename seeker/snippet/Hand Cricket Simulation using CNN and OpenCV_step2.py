#date: 2022-05-27T16:58:13Z
#url: https://api.github.com/gists/0b2da0666bfab434aa648d073cb0cebf
#owner: https://api.github.com/users/Abhinav-Bandaru

import os
import cv2

directory = 'dataset'
images_as_numpyarr = []
labels = []

for folder in os.listdir(directory):
    path = os.path.join(directory, folder)
    for item in os.listdir(path):
        new_path = os.path.join(path, item)
        for imgg in os.listdir(new_path):
            img = cv2.imread(os.path.join(new_path, imgg))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (240, 240))
            images_as_numpyarr.append(img)
            if item=='none':
              labels.append(7)
            else:
              labels.append(int(item))
