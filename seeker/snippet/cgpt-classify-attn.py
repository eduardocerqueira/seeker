#date: 2023-02-21T16:48:40Z
#url: https://api.github.com/gists/2295d1c37eb9574bb8d56608b61a47b9
#owner: https://api.github.com/users/mypapit

from keras.layers import Input, Conv2D, BatchNormalization, Activation, MaxPooling2D, Flatten, Dense, Dropout, Reshape, Permute, multiply, Lambda
from keras.models import Model
from keras import backend as K

# Define the attention mechanism layer
def attention(x):
    f = Conv2D(filters=64, kernel_size=1, strides=1, padding='same')(x)
    g = Conv2D(filters=64, kernel_size=1, strides=1, padding='same')(x)
    h = Conv2D(filters=256, kernel_size=1, strides=1, padding='same')(x)
    
    f = Reshape((-1, 64))(f)
    g = Reshape((-1, 64))(g)
    h = Reshape((-1, 256))(h)
    
    s = Lambda(lambda x: K.batch_dot(x[1], K.permute_dimensions(x[0], (0, 2, 1))))([f, g])
    beta = Activation('softmax')(s)
    o = Lambda(lambda x: K.batch_dot(x[0], x[1]))([beta, h])
    o = Reshape((8, 8, -1))(o)
    o = Conv2D(filters=256, kernel_size=1, strides=1, padding='same')(o)
    o = BatchNormalization()(o)
    o = Activation('relu')(o)
    
    return o

# Define the deep neural network model with attention mechanism
input_shape = (224, 224, 3)
inputs = Input(shape=input_shape)
x = Conv2D(filters=64, kernel_size=3, strides=1, padding='same')(inputs)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = Conv2D(filters=64, kernel_size=3, strides=1, padding='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPooling2D(pool_size=2)(x)

x = attention(x)

x = Flatten()(x)
x = Dense(units=256)(x)
x = Dropout(rate=0.5)(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)

outputs = Dense(units=num_classes, activation='softmax')(x)

model = Model(inputs=inputs, outputs=outputs)
model.summary()
