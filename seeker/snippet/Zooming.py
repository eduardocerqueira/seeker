#date: 2023-09-07T17:05:32Z
#url: https://api.github.com/gists/69bc8389b1d58cf6e8b7d4f6e6964c7f
#owner: https://api.github.com/users/mohsinipk

import tensorflow as tf

def zoom_image(image, zoom_range=(1.0, 1.2)):
    return tf.keras.preprocessing.image.apply_affine_transform(
        image, zx=zoom_range[1], zy=zoom_range[0]
    )
