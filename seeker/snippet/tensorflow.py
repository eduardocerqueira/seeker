#date: 2024-11-13T16:56:55Z
#url: https://api.github.com/gists/b67906784075e4e98ae8877b72914d9a
#owner: https://api.github.com/users/docsallover

import tensorflow as tf

# Create a constant tensor
tensor1 = tf.constant([[1, 2], [3, 4]])

# Create a variable tensor
tensor2 = tf.Variable([[5, 6], [7, 8]])

# Basic operations
tensor3 = tensor1 + tensor2
tensor4 = tf.matmul(tensor1, tensor2)

print(tensor3)
print(tensor4)