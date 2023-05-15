#date: 2023-05-15T16:54:34Z
#url: https://api.github.com/gists/fe7eeabdb6e2c8641c1d10dd6d18fd52
#owner: https://api.github.com/users/makslevental

import torch
from torch import nn


goofy_lib = torch.library.Library("goofy", "DEF")


class Kernel1(nn.Module):
    @staticmethod
    def forward(_3: torch.Tensor, _arg0: torch.Tensor, _5: torch.Tensor):
        return _5


class Kernel3(nn.Module):
    @staticmethod
    def forward(_1: torch.Tensor, _5: torch.Tensor, _7: torch.Tensor):
        return _7


goofy_lib.define("Kernel1(Tensor _3, Tensor _arg0, Tensor _5) -> Tensor")
goofy_lib.impl("Kernel1", Kernel1.forward)
goofy_lib.define("Kernel3(Tensor _1, Tensor _5, Tensor _7) -> Tensor")
goofy_lib.impl("Kernel3", Kernel3.forward)


class Forward(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, _1, _3, _arg0):
        _5 = torch.empty(1, 8, 14, 14)
        _5 = torch.ops.goofy.Kernel1(_3=_3, _arg0=_arg0, _5=_5)
        _7 = torch.empty(1, 2, 4, 6)
        _7 = torch.ops.goofy.Kernel3(_1=_1, _5=_5, _7=_7)
        return _7.sum()


if __name__ == "__main__":
    model = Forward()
    scripted = torch.jit.script(model)
    print(scripted.graph)
