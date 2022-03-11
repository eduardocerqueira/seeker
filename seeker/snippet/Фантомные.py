#date: 2022-03-11T17:10:11Z
#url: https://api.github.com/gists/28a4a49f957e4609de1fbe7d8a93461c
#owner: https://api.github.com/users/Mitlyy

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from numpy import *

Earkost = 190 #яркость (чем больше, тем ярче)

# def scale(X, x_min, x_max):
#     nom = (X-X.min(axis=0))*(x_max-x_min)
#     denom = X.max(axis=0) - X.min(axis=0)
#     denom[denom==0] = 1
#     return x_min + nom/denom


B_0 = np.mat(pd.read_csv('kNigga.csv',         #Название таблицы
                                delimiter=";",
                                header=None))   

G = 0
img_0 = np.zeros([128, 128])
img_2 = np.zeros([128, 128])
for i in range(2, 150): #колво картинок
    temp = Image.open('Masks/%d.png' % i) #папка с кратинками
    img2 = temp.convert('1')
    img = asarray(img2)
    img_0 += img
    G += B_0[i-2, 1]
    img_2 += img*B_0[i-2, 1]
print(G)
plt.imshow(Image.fromarray(Earkost*((img_2/i)-(1/i * G * 1/i * img_0))))
plt.show()
print(abs((img_2/i)-(1/i * G * 1/i * img_0)))

