#date: 2021-09-02T16:57:43Z
#url: https://api.github.com/gists/b3b4cb0aaf5d9cbec7066801fde8ce13
#owner: https://api.github.com/users/aswinvk28

def read_single_tfrecord_sample(sample):
    feature_map = {
      'class': tf.io.FixedLenFeature([], dtype=tf.int64),
      'image': tf.io.FixedLenFeature([], dtype=tf.string, default_value='')
    }
    example = tf.io.parse_single_example(sample, feature_map)
    image = tf.image.decode_jpeg(example['image'], channels=3)
    label = tf.cast(example['class'], tf.int32)
    return image, label

def read_single_test_data(sample):
    feature_map = {
      'id': tf.io.FixedLenFeature([], dtype=tf.string, default_value=''),
      'image': tf.io.FixedLenFeature([], dtype=tf.string, default_value='')
    }
    example = tf.io.parse_single_example(sample, feature_map)
    image = tf.image.decode_jpeg(example['image'], channels=3)
    identifier = example['id']
    return image, identifier

def load_train_dataset(filenames):
    raw_dataset = tf.data.TFRecordDataset(filenames)
    datasets = raw_dataset.map(read_single_tfrecord_sample)
    return datasets

def load_test_dataset(filenames):
    raw_dataset = tf.data.TFRecordDataset(filenames)
    datasets = raw_dataset.map(read_single_test_data)
    return datasets

def load_validation_dataset(filenames):
    raw_dataset = tf.data.TFRecordDataset(filenames)
    datasets = raw_dataset.map(read_single_tfrecord_sample)
    return datasets