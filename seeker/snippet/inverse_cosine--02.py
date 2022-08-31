#date: 2022-08-31T17:06:30Z
#url: https://api.github.com/gists/4fc2e00c2cc3047ca75223e850ecbb1e
#owner: https://api.github.com/users/rahulremanan

import math, tensorflow as tf
def tf_degrees(inp): return (tf.constant(180,dtype=tf.float32)/tf.constant(math.pi))*inp

tf_degrees(tf.math.acos(b/c))