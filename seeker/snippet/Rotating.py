#date: 2023-09-07T16:54:45Z
#url: https://api.github.com/gists/c3921cda24c0496604620554e55a0f2a
#owner: https://api.github.com/users/mohsinipk

import tensorflow as tf

def rotate_image(image, angle):
    return tf.keras.preprocessing.image.apply_affine_transform(
        image, theta=angle
    )