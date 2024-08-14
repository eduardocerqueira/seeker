#date: 2024-08-14T18:39:55Z
#url: https://api.github.com/gists/33b77e8d9a46ae2ff86a4e53df31ae5e
#owner: https://api.github.com/users/justinchuby

import torch

class UpsampleModel(torch.nn.Module):
    def forward(self, x):
        return torch.nn.functional.upsample_bilinear(x, scale_factor=2)


model = UpsampleModel()
ep = torch.export.export(model, (torch.randn(1, 3, 224, 224),))
ep.run_decompositions()
print(ep)
