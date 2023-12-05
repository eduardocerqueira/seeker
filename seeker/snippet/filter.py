#date: 2023-12-05T17:03:56Z
#url: https://api.github.com/gists/e4f275e0301e05c4a55adf6c5f1b0891
#owner: https://api.github.com/users/k-hkrachenfels


import numpy as np

values = np.array([1,2,3,4,5,6])
gt4int = np.where(values > 4,1,0)
gt4bool = np.where(values > 4,True,False)
print(gt4int,gt4bool)
print(values[values>4])

# Ausgaben
# [0 0 0 0 1 1] [False False False False  True  True]
# [5 6]
