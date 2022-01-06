#date: 2022-01-06T17:17:05Z
#url: https://api.github.com/gists/4f8c99a51220661db8391208497d30f6
#owner: https://api.github.com/users/FradSer

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches
import matplotlib.text as text

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

img = mpimg.imread('./image2.png')     
gray = rgb2gray(img)

x = np.arange(1024+1)[1:]

final_x = []

# x is some generator
for item in x:
    final_x.append(x)

final_y = np.rot90(final_x,3)

vb_x = np.sum((gray)*final_x) / np.sum(gray)
vb_y = np.sum((gray)*final_y) / np.sum(gray)

fig, ax = plt.subplots()

ax.add_patch(patches.Rectangle((0.45*1024, 0.45*1024), 0.1*1024, 0.1*1024, linewidth=1, edgecolor='g', facecolor='none'))
ax.text(vb_x/1024, 0.98 - vb_y/1024, '(' + '{:.1f}'.format(vb_x) + ', ' + '{:.1f}'.format(vb_y) + ')', horizontalalignment='left',verticalalignment='top', transform=ax.transAxes, color='r')

plt.imshow(gray, cmap=plt.get_cmap('gray'), vmin=0, vmax=1)
plt.plot(vb_x,vb_y,color='red', marker='o')

plt.show()