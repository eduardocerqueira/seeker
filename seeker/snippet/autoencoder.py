#date: 2025-07-11T17:08:50Z
#url: https://api.github.com/gists/2f6037c3c8fa4a5f9c09a5e9d38e8ffc
#owner: https://api.github.com/users/Xe-n00n

import tensorflow as tf
from tensorflow.keras import layers, Model, Input

input_length = 30
embedding_dim = 128
latent_dim = 32
num_classes = 183
#decoder

inputs = Input(shape=(input_length,), name='input')
x = layers.Embedding(input_dim=2000, output_dim=embedding_dim, input_length=input_length)(inputs)
x = layers.Conv1D(32, 3, activation='relu', padding='same')(x)
x = layers.MaxPooling1D(2)(x)
x = layers.Conv1D(64, 3, activation='relu', padding='same')(x)
encoded = layers.GlobalAveragePooling1D()(x)

#decoder
x = layers.Dense((input_length // 2) * 64, activation='relu')(encoded)
x = layers.Reshape((input_length // 2, 64))(x)
x = layers.Conv1D(32, 3, activation='relu', padding='same')(x)
decoded = layers.UpSampling1D(2)(x)
decoded = layers.Conv1D(1, 3, activation='sigmoid', padding='same', name='reconstruction')(decoded)

#Classification  
decoder_flat = layers.Flatten()(decoded)
merged = layers.Concatenate()([encoded, decoder_flat])
x = layers.Dense(128, activation='relu')(merged)
x = layers.Dropout(0.3)(x)
classification_output = layers.Dense(num_classes, activation='softmax', name='classification')(x)

model = Model(inputs=inputs, outputs= classification_output) 
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()
