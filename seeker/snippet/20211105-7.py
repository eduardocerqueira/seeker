#date: 2021-11-05T16:48:23Z
#url: https://api.github.com/gists/c03f4d5b5b059de8fad15d5d7d4c4547
#owner: https://api.github.com/users/thigm85

# Create a description of the features.
feature_description = {
    'id': tf.io.FixedLenFeature([1], tf.string, default_value=''),
    'labels': tf.io.FixedLenSequenceFeature([], tf.int64, default_value=0, allow_missing=True),
    'mean_audio': tf.io.FixedLenFeature([128], tf.float32, default_value=[0.0] * 128),    
    'mean_rgb': tf.io.FixedLenFeature([1024], tf.float32, default_value=[0.0] * 1024),
}

def _parse_function(example_proto):
  # Parse the input `tf.train.Example` proto using the dictionary above.
  return tf.io.parse_single_example(example_proto, feature_description)