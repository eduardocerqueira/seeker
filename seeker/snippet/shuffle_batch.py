#date: 2021-10-12T16:56:39Z
#url: https://api.github.com/gists/c206b740c58698a852658bb042537c6a
#owner: https://api.github.com/users/ashrafdasa

import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def printDs(ds, quanitiy = 5):
    print('---------------------')
    for example in ds.take(quanitiy):
        print(example)



data = list(range(100))
ds = tf.data.Dataset.from_tensor_slices(data)
print(type(ds))

printDs(ds)
## SUFFLE
shuffled = ds.shuffle(buffer_size=5)
printDs(shuffled,50)

shuffled = ds.shuffle(buffer_size=1)
printDs(shuffled,50)
shuffled = ds.shuffle(buffer_size=5)
printDs(shuffled,10)

printDs(shuffled,10)

shuffled = ds.shuffle(buffer_size=5, reshuffle_each_iteration = False)
printDs(shuffled,10)

printDs(shuffled,10)

# batching
batched = ds.batch(10)
printDs(batched)

batched = ds.batch(14)
printDs(batched,10)

batched = ds.batch(14, drop_remainder=True)
printDs(batched,10)

Shuffle_batched = ds.batch(14, drop_remainder=True).shuffle(buffer_size=5)
printDs(Shuffle_batched,10)

Batch_shuffled = ds.shuffle(buffer_size=5).batch(14, drop_remainder=True)
printDs(Batch_shuffled,10)

shuffle_Batch_shuffled = ds.shuffle(buffer_size=5).batch(14, drop_remainder=True).shuffle(buffer_size=50)
printDs(shuffle_Batch_shuffled,10)

myFinalDs = ds.shuffle(buffer_size=5, reshuffle_each_iteration=False).batch(14, drop_remainder=True).shuffle(buffer_size=50, reshuffle_each_iteration=False)
printDs(myFinalDs,10)
printDs(myFinalDs,10)



shuffle_batch = ds.shuffle(100).batch(10)
printDs(shuffle_batch,10)
printDs(shuffle_batch,10)
