#date: 2023-03-31T17:05:20Z
#url: https://api.github.com/gists/31ed6f5d3b1efbaf50c1ae6a4ce7e119
#owner: https://api.github.com/users/CharlemagneBrain

'''
If you are interested in what inside your tfrecords.
Your can print it out like below.
'''

def get_tfrecords_features():
    return {
        'x': tf.FixedLenFeature([4096], tf.float32),
        'y': tf.FixedLenFeature([4096], tf.float32),
        'z': tf.FixedLenFeature([4096], tf.float32),
        'norm_x': tf.FixedLenFeature([4096], tf.float32),
        'norm_y': tf.FixedLenFeature([4096], tf.float32),
        'norm_z': tf.FixedLenFeature([4096], tf.float32),
        'r': tf.FixedLenFeature([4096], tf.float32),
        'g': tf.FixedLenFeature([4096], tf.float32),
        'b': tf.FixedLenFeature([4096], tf.float32),
        'label': tf.FixedLenFeature([4096], tf.int64)
        }


def load_tfrecords(tfrecords_filepath):
    '''
    Input:
        Path to a tfrecord file

    Returns:
        Tensor
    '''
    items = []
    labels = []
    print("Loading %s" % tfrecords_filepath)
    with tf.Session() as sess:
        for serialized_example in tf.python_io.tf_record_iterator(tfrecords_filepath):

            data, label = feature_retrieval(serialized_example)

            items.append(data)
            labels.append(label)
    print("Finished Loading %s" % tfrecords_filepath)
    return (tf.stack(items), tf.stack(labels))


def feature_retrieval(serialized_example):

    features = tf.parse_single_example(serialized_example, features=get_tfrecords_features())

    _x = tf.cast(features['x'], tf.float32)
    _y = tf.cast(features['y'], tf.float32)
    _z = tf.cast(features['z'], tf.float32)
    _norm_x = tf.cast(features['norm_x'], tf.float32)
    _norm_y = tf.cast(features['norm_y'], tf.float32)
    _norm_z = tf.cast(features['norm_z'], tf.float32)
    _r = tf.cast(features['r'], tf.float32)
    _g = tf.cast(features['g'], tf.float32)
    _b = tf.cast(features['b'], tf.float32)
    _label = tf.cast(features['label'], tf.int64)

    data = tf.transpose(
            tf.stack(
                [
                    _x, 
                    _y, 
                    _z, 
                    _norm_x, 
                    _norm_y, 
                    _norm_z, 
                    _r, 
                    _g, 
                    _b
                ])
        )
    label = tf.transpose(_label)
    return data, label
  
if __name__ == '__main__':
  data, label = load_tfrecords(tfrecords_filepath)