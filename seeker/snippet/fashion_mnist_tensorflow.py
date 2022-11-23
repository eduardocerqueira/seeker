#date: 2022-11-23T16:54:09Z
#url: https://api.github.com/gists/c7a854a47f5c225aa4996651c61904d6
#owner: https://api.github.com/users/linnil1

# Modified from https://github.com/tensorflow/docs/blob/master/site/en/tutorials/keras/classification.ipynb
import tensorflow as tf
import numpy as np

# gpu setup
gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# dataset
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
train_images = train_images / 255.0
test_images = test_images / 255.0

# model
model = tf.keras.Sequential(
    [
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(16384, activation="relu"),
        tf.keras.layers.Dense(16384, activation="relu"),
        tf.keras.layers.Dense(16384, activation="relu"),
        tf.keras.layers.Dense(16384, activation="relu"),
        tf.keras.layers.Dense(16384, activation="relu"),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(10),
    ]
)

# model parameters
model.compile(
    optimizer="adam",
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)
# train
model.fit(train_images, train_labels, batch_size=64, shuffle=True, epochs=30)
# evalaute
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print("\nTest accuracy:", test_acc)