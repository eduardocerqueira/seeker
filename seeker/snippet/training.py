#date: 2023-01-30T16:54:51Z
#url: https://api.github.com/gists/69203b58c5b5b4dfe8ebf7f1c43ed517
#owner: https://api.github.com/users/malizia-g

#pip install tensorflow

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Creazione del modello
model = keras.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Compilazione del modello
model.compile(optimizer='adam',
              loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Caricamento del dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train / 255.0
x_test = x_test / 255.0
x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]

# Addestramento del modello
model.fit(x_train, y_train, epochs=5)

# Valutazione del modello
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print('\nTest accuracy:', test_acc)


model.save('model.h5')