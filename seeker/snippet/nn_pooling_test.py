#date: 2024-04-11T17:10:53Z
#url: https://api.github.com/gists/cd3ba4dfd23db151d1a6fbf10db2ec30
#owner: https://api.github.com/users/clw5710

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras import initializers  

# KERAS MODEL
def nnPoolTest():
    inputs = keras.Input(shape=(5, 5, 5))
    convIn = layers.Conv2D(5, kernel_size=(3, 1), strides=1, padding='same', use_bias=False)
    avgpool = layers.AveragePooling2D(pool_size=1, strides=2) 
    zeropad_2a = layers.ZeroPadding2D(padding=((0,0),(0,5)), data_format="channels_first")
    relu   = layers.ReLU()

    initializer = initializers.HeNormal(seed=None)
    fcOut   = layers.Dense(5, input_shape=(5,), kernel_initializer=initializer, bias_initializer='zeros', use_bias=True, activation=None)
    flatten = layers.Flatten()


    z = convIn(inputs)
    z = avgpool(z)
    z = zeropad_2a(z)
    z = relu(z) 
    z = flatten(z)
    outputs = fcOut(z)
    return keras.Model(inputs=inputs, outputs=outputs)