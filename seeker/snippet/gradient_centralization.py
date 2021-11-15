#date: 2021-11-15T17:04:07Z
#url: https://api.github.com/gists/a7662d39929f7fed7644f0e0f7e8ad28
#owner: https://api.github.com/users/janaSunrise

# Imports
import tensorflow as tf
from keras import backend as K


# Define function for centralizing the gradients
def centralize_gradients(optimizer, loss, params):
    grads = []  # List to store the gradients

    for grad in K.gradients(loss, params):  # Iterate over gradients using the Keras Backend
        grad_len = len(grad.shape)  # Get the size of the gradient

        if grad_len > 1:
            axis = list(range(grad_len - 1))
            grad -= tf.reduce_mean(grad, axis=axis, keep_dims=True)  # Clip off the mean

        grads.append(grad)

    if None in grads:
        raise ValueError("""
        An operation has `None` for gradient. Please make sure that all of your ops have a
        gradient defined (That is, are differentiable). Common operations without gradients
        are: `K.argmax`, `K.round` and `K.eval`.
        """)

    # Clip gradients
    if hasattr(optimizer, "clipnorm") and optimizer.clipnorm > 0:
        # Get normalized gradients
        norm = K.sqrt(sum([K.sum(K.square(g)) for g in grads]))
        
        # Clip normalized gradients
        grads = [
            tf.keras.optimizers.clip_norm(g, optimizer.clipnorm, norm) for g in grads
        ]

    # Do the final clip and update the list
    if hasattr(optimizer, "clipvalue") and optimizer.clipvalue > 0:
        grads = [K.clip(g, -optimizer.clipvalue, optimizer.clipvalue) for g in grads]

    return grads


# Function to apply gradient centralization to optimizers.
def apply_gradient_centralization(optimizer):
    """
    adam = tf.keras.optimizers.Adam(...)
    adam.get_gradients = apply_gradient_centralization(adam)
    """
    def get_centralized_gradients(loss, params):
        return centralize_gradients(optimizer, loss, params)

    return get_centralized_gradients
