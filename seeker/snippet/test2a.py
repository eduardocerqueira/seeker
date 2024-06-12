#date: 2024-06-12T16:49:58Z
#url: https://api.github.com/gists/99ba2eac937f62130eb87f3c5ab95e7d
#owner: https://api.github.com/users/plopresti

#!/usr/bin/env python

import tensorflow as tf

@tf.function
def f(x, y):
  test = tf.Module()
  test.output = {
      "foo": x ** 2 + y,
      "bar": x ** 2 - y,
  }
  return test.output

x = tf.constant([2, 3])
y = tf.constant([3, -2])
f(x, y)

print('no bug!')
