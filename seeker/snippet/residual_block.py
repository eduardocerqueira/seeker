#date: 2023-06-15T17:06:13Z
#url: https://api.github.com/gists/858b01e381e138aa6b882a796b7c384d
#owner: https://api.github.com/users/jacksalici


import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    inplanes = 0
    planes = 0
    stride = 0

    def __init__(self, inplanes, planes, stride):
        super().__init__()

        self.inplanes = inplanes
        self.planes = planes
        self.stride = stride


        self.conv1 = nn.Conv2d(inplanes, planes, stride=stride, kernel_size=(3,3), bias=0)
        self.conv2 = nn.Conv2d(planes, planes,  kernel_size=(3,3), bias=0)


    def g(self, x):
      if self.inplanes != self.planes or self.stride > 1:
          conv3 = nn.Conv2d(self.inplanes, self.planes, stride=self.stride, kernel_size=(1,1), bias=0)
          m = nn.BatchNorm2d(x[1], track_running_stats=True)
          y = F.relu(m(conv3(x)))
      else: 
          y = x
      return y

    def forward(self, x):
        x1 = self.conv1(x)
        m = nn.BatchNorm2d(x1[1], track_running_stats=True)
        x1 = F.relu(m(x1))

        x1 = self.conv2(x1)
        m = nn.BatchNorm2d(x[1], track_running_stats=True)
        
        return F.relu(x1+self.g(x))