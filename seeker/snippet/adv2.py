#date: 2023-03-22T17:04:36Z
#url: https://api.github.com/gists/701239f3af79cd03da864b76cd9a816a
#owner: https://api.github.com/users/corrosivelogic

import torch
import torch.nn as nn
from torchvision.models import resnet50

class Normalize(nn.Module):
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.mean = torch.Tensor(mean)
        self.std = torch.Tensor(std)
    def forward(self, x):
        return (x - self.mean.type_as(x)[None,:,None,None]) / self.std.type_as(x)[None,:,None,None]


norm = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

model = resnet50(pretrained=True)
model.eval();
# form predictions
pred = model(norm(pig_tensor))
print(pred.max(dim=1)[1].item())