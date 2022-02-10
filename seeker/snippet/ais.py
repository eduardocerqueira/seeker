#date: 2022-02-10T17:01:26Z
#url: https://api.github.com/gists/ff69d3bc52506244e0d5dee50f844d47
#owner: https://api.github.com/users/Medhashaharkar

from keras import layers
from keras import models
from keras import optimizers
import numpy as np
import pandas as pd

data = pd.read_csv("/kaggle/input/beginners-classification-dataset/classification.csv")
data.head()
data.shape
x_train = data.loc[0:207,['age','interest']]
x_val = data.loc[208:267,['age','interest']]
x_test = data.loc[267:296,['age','interest']]
y_train = data.loc[0:207,'success']
y_val = data.loc[208:267,'success']
y_test = data.loc[267:296,'success']

model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(2,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop',
              loss = 'binary_crossentropy',
              metrics = ['accuracy'])
history = model.fit(x_train,
                    y_train,
                    epochs = 6,
                    batch_size = 16,
                   validation_data=(x_val, y_val))
results = model.evaluate(x_test, y_test)

