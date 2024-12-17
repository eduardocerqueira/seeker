#date: 2024-12-17T16:49:28Z
#url: https://api.github.com/gists/72571b7201855b25b0f5cea517d297e2
#owner: https://api.github.com/users/docsallover

import torch

x = torch.tensor([2.0], requires_grad=True)
y = x**2
y.backward()
print(x.grad)  # Output: tensor([4.])