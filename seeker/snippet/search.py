#date: 2021-10-04T17:18:15Z
#url: https://api.github.com/gists/364f6baf53c005156f32dbc446c81fe8
#owner: https://api.github.com/users/danielvarga

# https://en.wikipedia.org/wiki/Mutually_unbiased_bases#The_problem_of_finding_a_maximal_set_of_MUBs_when_d_=_6
# code based on question
# https://stackoverflow.com/questions/64821151/notfounderror-when-using-a-tensorflow-optimizer-on-complex-variables-on-a-gpu

import tensorflow as tf
import numpy as np


n = 3
m = 4

mat_r = tf.random.uniform([n * m, n], dtype=tf.float64)
mat_i = tf.random.uniform([n * m, n], dtype=tf.float64)
mat = 2 * tf.complex(mat_r, mat_i) - 1

var = tf.Variable(mat, trainable=True)

identity = tf.eye(n, dtype=tf.complex128)

def closeness(a, b):
    return tf.reduce_sum(tf.abs(a - b) ** 2)

# Return the squared norm of this matrix as the loss function
def lossFn():
    us = []
    terms = []
    for i in range(m):
        u = var[i * n: (i + 1) * n]
        us.append(u)
        terms.append(closeness(tf.transpose(u, conjugate=True) @ u, identity))

    target = tf.ones((n, n), dtype=tf.float64) / n ** 0.5
    for i in range(m):
        for j in range(i + 1, m):
            prod = tf.transpose(us[i], conjugate=True) @ us[j]
            terms.append(closeness(tf.abs(prod), target))
    return sum(terms)

opt = tf.keras.optimizers.SGD(learning_rate=0.01)

for iteration in range(1000):
    with tf.GradientTape() as tape:
        loss = lossFn()
    grads = tape.gradient(loss, [var])
    opt.apply_gradients(zip(grads, [var]))
    if iteration % 100 == 0:
        print(iteration, loss.numpy())

u0 = var[:n, :n].numpy()
u1 = var[n: 2*n, :n].numpy()

print("----\nA† A")
print(np.conjugate(u0.T) @ u0)
print("----\n√n |A† B|")
print(n ** 0.5 * np.abs(np.conjugate(u0.T) @ u1))
print("----\nA")
print(u0)
print("----\nB")
print(u1)
