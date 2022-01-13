#date: 2022-01-13T17:19:57Z
#url: https://api.github.com/gists/382ef8c23622dcfd4c878cc403fe6c74
#owner: https://api.github.com/users/pemoi1982jpm

import tensorflow as tf
n_blocks = 4

@tf.function(jit_compile=True)
def myfunc(segment_ids, vector):
    value = tf.ones_like(segment_ids)
    size_blocks = tf.math.unsorted_segment_sum(value, segment_ids, num_segments=n_blocks)
    offsets = tf.concat([[0], tf.math.cumsum(size_blocks)], axis=-1)
    blocks_list = []
    for i in range(n_blocks):
        block = vector[offsets[i]:offsets[i+1]]
        block = tf.einsum('i,j -> ij', block, tf.transpose(block))
        blocks_list.append(tf.linalg.LinearOperatorFullMatrix(block))
    matrix = tf.linalg.LinearOperatorBlockDiag(blocks_list)
    
    return matrix.matvec(vector)

for i in range(3):
    segment_ids = tf.constant([0, 0, 1, 1, 2, 3])
    vector = tf.ones([6], dtype=tf.float64)
    print(myfunc(segment_ids, vector))