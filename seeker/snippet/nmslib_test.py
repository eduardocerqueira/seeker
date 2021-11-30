#date: 2021-11-30T17:09:37Z
#url: https://api.github.com/gists/d5ecc5ea8f18fbad510312e6bd2c30b2
#owner: https://api.github.com/users/liusy182

import nmslib
import numpy as np
import os, psutil

dim = 128
num_elements = 100000

rs = np.random.RandomState(123)
data = np.float32(rs.random((num_elements, dim)))

index = nmslib.init(method='hnsw', space='l2')
index.addDataPointBatch(data)
index.createIndex()

process = psutil.Process(os.getpid())
print(process.memory_info())

# neighbours = index.knnQueryBatch(data[:5], k=5, num_threads=4)
# print('neighbours', len(neighbours))
# for l in neighbours:
#     print('label', l[0])
#     print('distance', l[1])
