#date: 2021-10-26T16:59:55Z
#url: https://api.github.com/gists/0c8de83b9d4b4c6a0e2e1fa1e9989028
#owner: https://api.github.com/users/b0noI

import torch

torch.manual_seed(2021)

w = torch.rand(1, requires_grad=True, dtype=torch.float64)
b = torch.rand(1, requires_grad=True, dtype=torch.float64)

def model(X):
  return X * w + b