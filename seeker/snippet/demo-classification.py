#date: 2022-12-15T16:48:53Z
#url: https://api.github.com/gists/ef50ac6be23e9c09437a47e94b8ef89e
#owner: https://api.github.com/users/SpecLad

#!/usr/bin/env python

import logging
import os

import cvat_sdk
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms

from cvat_sdk.pytorch import TaskVisionDataset, ExtractSingleLabelIndex


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 5*5 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square, you can specify with a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def run_training(
    trainset: torch.utils.data.Dataset,
    testset: torch.utils.data.Dataset,
):
    batch_size = 4
    num_workers = 0

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=num_workers)

    logging.info('Created data loaders')

    net = Net()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    logging.info('Started Training')

    for epoch in range(2):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                logging.info(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

    logging.info('Finished training')

    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            # calculate outputs by running images through the network
            outputs = net(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    logging.info('Finished testing')

    print(f'Accuracy of the network on the {len(testset)} test images: {correct / total:.2%}')
    
# ---------------------------------------------
# Everything above this point is from the PyTorch tutorial

def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info('Starting...')

    with cvat_sdk.make_client(
        'app.cvat.ai', credentials=(os.getenv("CVAT_USER"), os.getenv("CVAT_PASS"))
    ) as client:
        logging.info('Created the client')

        trainset = TaskVisionDataset(client, 34040,
            transform=transforms.Compose([
                transforms.Grayscale(),
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ]),
            target_transform=ExtractSingleLabelIndex(),
        )

        logging.info('Created the training dataset')

        testset = TaskVisionDataset(client, 47905, transforms=trainset.transforms)

        logging.info('Created the testing dataset')

        run_training(trainset, testset)

if __name__ == '__main__':
    main()
