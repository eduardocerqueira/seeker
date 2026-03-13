#date: 2026-03-13T17:31:37Z
#url: https://api.github.com/gists/553c1a07b2f6f6dc7389a054e0ea7799
#owner: https://api.github.com/users/shunting314

import torch

torch.set_default_device("cuda")

@torch.compile
def f(x, w):
    y = torch.nn.functional.rms_norm(x, x.shape[-1:], weight=None)
    return y * w

B, H, D = 32 * 1024, 2, 1024
x = torch.randn(B, H, D, requires_grad=True)
w = torch.randn(H, D, requires_grad=True)

f(x, w).sum().backward()
