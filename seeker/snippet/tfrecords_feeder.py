#date: 2023-03-31T17:05:20Z
#url: https://api.github.com/gists/31ed6f5d3b1efbaf50c1ae6a4ce7e119
#owner: https://api.github.com/users/CharlemagneBrain

'''
When actually using it, we will feed it as iterator that would not take a lot of memories of your machine.
'''

def tfrecords_to_dataset(handle):
    '''
    Note: We can't excplicitly select what data to use when using tf.data.Dataset
    Hence we will separate it manually like 0.2 or so.
    '''
    files = [ PATH_TO_EACH_TFRECORD ]

    TEST_PERCENT = 0.2

    train_dataset = tf.data.TFRecordDataset(files[int(TEST_PERCENT*len(files)): ])
    train_dataset = train_dataset.map(feature_retrieval)
    train_dataset = train_dataset.apply(tf.contrib.data.batch_and_drop_remainder(BATCH_SIZE))

    test_dataset = tf.data.TFRecordDataset(files[ : int(TEST_PERCENT*len(files))])
    test_dataset = test_dataset.map(feature_retrieval)
    test_dataset = test_dataset.apply(tf.contrib.data.batch_and_drop_remainder(BATCH_SIZE))

    iterator = tf.data.Iterator.from_string_handle(
            handle,
            train_dataset.output_types,
            train_dataset.output_shapes)
    
    next_elem = iterator.get_next()

    train_init_iter = train_dataset.make_initializable_iterator()
    test_init_iter = test_dataset.make_initializable_iterator()

    return next_elem, train_init_iter, test_init_iter
  
if __name__ == '__main__':
  handle_pl = tf.placeholder(tf.string, shape=[])
  next_value, train_init_iter, test_init_iter = tfrecords_to_dataset(handle_pl)
  pointclouds_pl, labels_pl = next_value
  
  loss = YOUR_MODEL(pointclouds_pl, labels_pl)
  
  # Training
  with tf.Session() as sess:
    training_handle = sess.run(train_init_iter.string_handle())
    sess.run(train_init_iter.initializer)
    sess.run(loss, feed_dict={'handle_pl' = training_handle})
    
  with tf.Session() as sess:
    testing_handle = sess.run(test_init_iter.string_handle())
    sess.run(test_init_iter.initializer)
    sess.run(loss, feed_dict={'handle_pl' = testing_handle})