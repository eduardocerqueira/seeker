#date: 2021-11-16T17:05:53Z
#url: https://api.github.com/gists/d8120f7cbc1a80adf54f9991a4b488e1
#owner: https://api.github.com/users/arpithaupd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms, datasets

trainset = datasets.MNIST('', train=True, download=True, 
                       transform=transforms.Compose([
                                transforms.ToTensor()
                            ]))
testset = datasets.MNIST('', train=False, download=True, 
                       transform=transforms.Compose([
                                transforms.ToTensor()
                            ]))


trainloader  = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=True, pin_memory=True)
testloader  = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=True, pin_memory=True)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # convolution 
        self.conv1 = nn.Conv2d(1, 64, 5)  # input channel=1, output channel=64, kernal size = 3*3
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(64, 32, 5)
        # fully connected 
        self.fc1 = nn.Linear(32 * 4 * 4, 128) # 32 channel, 4 * 4 size(經過Convolution部分後剩4*4大小)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 64)
        
    def forward(self, x):
        # state size. 28 * 28(input image size = 28 * 28)
        x = self.pool(F.relu(self.conv1(x)))
        # state size. 12 * 12
        x = self.pool(F.relu(self.conv2(x)))
        # state size. 4 * 4
        x = x.view(-1, 32 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
  
        return F.log_softmax(x, dim=1)
      
net = Net() # inital network

optimizer = optim.Adam(net.parameters(), lr=0.001)  # create a Adam optimizer

net.train()
epochs = 2
for epoch in range(epochs):
    running_loss = 0.0
    for i, data in enumerate(trainloader):
        X, y = data

        # training process
        optimizer.zero_grad()    # clear the gradient calculated previously
        predicted = net(X)  # put the mini-batch training data to Nerual Network, and get the predicted labels
        loss = F.nll_loss(predicted, y)  # compare the predicted labels with ground-truth labels
        loss.backward()      # compute the gradient
        optimizer.step()     # optimize the network
        running_loss += loss.item()
        if i % 100 == 99:    # print every 1000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 100))
            running_loss = 0.0
            
"""
model.train()" and "model.eval()" activates and deactivates Dropout and BatchNorm, so it is quite important. 
"with torch.no_grad()" only deactivates gradient calculations, but doesn't turn off Dropout and BatchNorm.
Your model accuracy will therefore be lower if you don't use model.eval() when evaluating the model.
"""
net.eval()

correct = 0
total = 0
with torch.no_grad():
    for data in trainloader:
        X, y = data
        output = net(X)
        correct += (torch.argmax(output, dim=1) == y).sum().item()
        total += y.size(0)

print(f'Training data Accuracy: {correct}/{total} = {round(correct/total, 3)}')

# Evaluation the testing data
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        X, y = data
        output = net(X)
        correct += (torch.argmax(output, dim=1) == y).sum().item()
        total += y.size(0)

print(f'testing data Accuracy: {correct}/{total} = {round(correct/total, 3)}')