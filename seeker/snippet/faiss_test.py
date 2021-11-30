#date: 2021-11-30T17:09:37Z
#url: https://api.github.com/gists/d5ecc5ea8f18fbad510312e6bd2c30b2
#owner: https://api.github.com/users/liusy182

import faiss
import numpy as np
import os, psutil

dim = 128
num_elements = 100000

rs = np.random.RandomState(123)
data = np.float32(rs.random((num_elements, dim)))
index = faiss.IndexHNSWFlat(dim, 16)
index.add(data)

process = psutil.Process(os.getpid())
print(process.memory_info())

# distances, labels = index.search(data[:5], 5)
# print('labels', len(labels))
# for l in labels:
#     print(l)
# print('distances', len(distances))
# for d in distances:
#     print(d)
