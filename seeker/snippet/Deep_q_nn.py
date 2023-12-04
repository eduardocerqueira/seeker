#date: 2023-12-04T17:02:35Z
#url: https://api.github.com/gists/2aee350f5acd7a0988a918b0b8296f99
#owner: https://api.github.com/users/rafea25

import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

#classes specific for snake game

class CNN_DQN(nn.Module):
    def __init__(self, game_dimension):
        input_channel = 1
        output_channel = 1
        super(CNN_DQN, self).__init__()
        self.layer1 = nn.Conv2d(input_channel, output_channel, kernel_size=5, stride=3, padding=1)
        #self.layer2 = nn.Linear(output_channel * game_dimension * game_dimension, 128)
        self.layer2 = nn.Linear(9, 128)
        self.layer3 = nn.Linear(128, 64)
        self.layer4 = nn.Linear(64, 4)
        #so since we have 4 outputs we're getting a 1D array of 4 values, each value represents the q value of an action we can take
        
    def forward(self, x):
        #x = torch.stack(x)
        x = F.relu(self.layer1(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = self.layer4(x)
        return x.squeeze()
    
class CNN_DQN_V3(nn.Module):
    def __init__(self, game_dimension):
        super(CNN_DQN_V3, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(6400, 512)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(512, 4) 
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        #print("Shape after conv1:", x.shape)
        x = self.conv2(x)
        x = self.relu2(x)
        #print("Shape after conv2:", x.shape)
        x = self.flatten(x)
        #print("Shape after flatten:", x.shape)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        return x
