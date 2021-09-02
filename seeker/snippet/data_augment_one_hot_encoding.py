#date: 2021-09-02T16:54:49Z
#url: https://api.github.com/gists/d907b7a91ae113b3b3494d699b0a8c5f
#owner: https://api.github.com/users/aswinvk28

def data_augment(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_saturation(image, lower=0, upper=2)
    return image, label

# label is int32
def one_hot_encoded(image, label):
    value = tf.constant(1)
    value = tf.expand_dims(value, 0)
    left_pad = tf.expand_dims(label, 0)
    right_pad = tf.expand_dims(tf.constant(len(CLASSES)) - label - 1, 0)
    one_hot_encoded = tf.pad(value, tf.expand_dims(tf.transpose(tf.concat([left_pad, right_pad], 0)), 0))
    return image, one_hot_encoded