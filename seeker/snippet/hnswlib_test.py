#date: 2021-11-30T17:09:37Z
#url: https://api.github.com/gists/d5ecc5ea8f18fbad510312e6bd2c30b2
#owner: https://api.github.com/users/liusy182

import hnswlib
import numpy as np
import os, psutil

dim = 128
num_elements = 100000

rs = np.random.RandomState(123)
data = np.float32(rs.random((num_elements, dim)))
ids = np.arange(num_elements)

p = hnswlib.Index(space = 'l2', dim = dim)
p.init_index(max_elements = num_elements, ef_construction = 200, M = 16)
p.add_items(data, ids)
p.set_ef(50) # ef should always be > k

process = psutil.Process(os.getpid())
print(process.memory_info())

# labels, distances = p.knn_query(data[:5], k = 5)
# print('labels', len(labels))
# for l in labels:
#     print(l)
# print('distances', len(distances))
# for d in distances:
#     print(d)
