#date: 2021-09-23T16:58:17Z
#url: https://api.github.com/gists/26aed9a46518d61ddfe194a41ef64c10
#owner: https://api.github.com/users/GuldenizBektas

import numpy as np

def gaussian_pdf(x, mean, std_dev):
  return 1 / (std_dev * np.sqrt(2 * np.pi)) * np.exp(-1 * (x - mean)**2 / (2 * std_dev**2))

def kl_divergence(p, q):
  # args:
  #   p -> gaussian-1, tupple(mean, std)
  #   q -> gaussian-2, tupple(mean, std)

  # return:
  #   a floating point number that denotes D(p||q)

  p_dist = np.random.normal(p[0], p[1], 1000)
  q_dist = np.random.normal(q[0], q[1], 1000)

  p = gaussian_pdf(p_dist, p[0], p[1])
  q = gaussian_pdf(q_dist, q[0], q[1])
  d = np.sum(p * np.maximum(np.log(p / q), 0))

  return d