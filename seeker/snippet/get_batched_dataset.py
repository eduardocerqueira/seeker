#date: 2021-09-02T16:57:43Z
#url: https://api.github.com/gists/b3b4cb0aaf5d9cbec7066801fde8ce13
#owner: https://api.github.com/users/aswinvk28

def get_batched_dataset(filenames, train=False, augment=False):
    function = load_train_dataset if train else load_test_dataset
    dataset = function(filenames)
    dataset = dataset.cache() # This dataset fits in RAM
    if augment:
        # Best practices for Keras:
        # Training dataset: repeat then batch
        # Evaluation dataset: do not repeat
        dataset = dataset.repeat()
        dataset = dataset.map(data_augment, num_parallel_calls=AUTO)
        dataset = dataset.shuffle(2000)
    if train:
        dataset = dataset.map(one_hot_encoded)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)
    # should shuffle too but this dataset was well shuffled on disk already
    return dataset

train_dataset = get_batched_dataset(train_filenames, train=True, augment=True)
val_dataset = get_batched_dataset(val_filenames, train=True, augment=False)
test_dataset = get_batched_dataset(val_filenames, train=False, augment=False)