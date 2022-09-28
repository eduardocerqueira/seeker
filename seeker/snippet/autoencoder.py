#date: 2022-09-28T17:13:46Z
#url: https://api.github.com/gists/23a7106796049d462a41b747c7c981b3
#owner: https://api.github.com/users/ph04

from mnist import MNIST
from PIL import Image
import jax.numpy as jnp
import numpy as np
import jax

STEP_SIZE = 0.0001

# mndata = MNIST("./mnist")
#
# images, labels = mndata.load_training()
#
# images_test, labels_test = mndata.load_testing()
#
# dataset = jnp.array(images)
#
# jnp.save("output", dataset)

def de_normalize(x):
    return x * 255

def generate_image(image_array):
    reshaped = np.array(image_array).reshape((28, 28))
    return Image.fromarray(reshaped)

def autoencoder(params, x):
    w1, b1, w2, b2 = params
    return w2 @ (w1 @ x + b1) + b2

def loss(params, x):
    y = autoencoder(params, x) 

    return jnp.square(y - x).mean()

@jax.jit
def step(params, x):
    batch_loss = lambda params, batch : jax.vmap(loss, in_axes=(None, 0))(params, batch).mean()

    loss_value, gradient = jax.value_and_grad(batch_loss)(params, x)

    new_params = [p - g * STEP_SIZE for p, g in zip(params, gradient)]

    return new_params, loss_value

data = jnp.load("output.npy")

data = data / 255

key = jax.random.PRNGKey(9999)

key, seed = jax.random.split(key)
w1 = jax.random.normal(seed, (20, 784)) * STEP_SIZE

key, seed = jax.random.split(key)
b1 = jax.random.normal(seed, (20, 1)) * STEP_SIZE

key, seed = jax.random.split(key)
w2 = jax.random.normal(seed, (784, 20)) * STEP_SIZE

key, seed = jax.random.split(key)
b2 = jax.random.normal(seed, (784, 1)) * STEP_SIZE

params = [w1, b1, w2, b2]

for _ in range(10_000):
    key, seed = jax.random.split(key)

    data = jax.random.permutation(seed, data)

    for i in range(0, 3750, 16):
        params, loss_value = step(params, data[i : i + 16])

        print(loss_value)