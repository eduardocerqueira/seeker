#date: 2022-11-23T16:54:09Z
#url: https://api.github.com/gists/c7a854a47f5c225aa4996651c61904d6
#owner: https://api.github.com/users/linnil1

# Modified from https://github.com/pranay414/Fashion-MNIST-Pytorch/blob/master/fashion_mnist.ipynb
import torch
from torch import nn
from torchvision import datasets, transforms
from tqdm import tqdm

# dataset
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)
trainset = datasets.FashionMNIST(
    ".keras/datasets/fashion-mnist/", download=True, train=True, transform=transform
)
testset = datasets.FashionMNIST(
    ".keras/datasets/fashion-mnist/", download=True, train=False, transform=transform
)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=64, shuffle=True, num_workers=4
)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=64, shuffle=True, num_workers=4
)

# model
model = nn.Sequential(
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 16384),
    nn.ReLU(),
    nn.Linear(16384, 16384),
    nn.ReLU(),
    nn.Linear(16384, 16384),
    nn.ReLU(),
    nn.Linear(16384, 16384),
    nn.ReLU(),
    nn.Linear(16384, 16384),
    nn.ReLU(),
    nn.Linear(16384, 16384),
    nn.ReLU(),
    nn.Linear(16384, 128),
    nn.ReLU(),
    nn.Linear(128, 10),
    nn.LogSoftmax(dim=1),
).cuda()

# model parameters
criterion = nn.NLLLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
epochs = 30

# train
train_losses, test_losses = [], []
for e in range(epochs):
    running_loss = 0
    model.train()
    for images, labels in tqdm(trainloader):
        images = images.cuda()
        labels = labels.cuda()
        # training step
        images = images.view(images.shape[0], -1)
        optimizer.zero_grad()
        output = model.forward(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    train_losses.append(running_loss / len(trainloader))

    # evaluate
    test_loss = 0
    accuracy = 0
    with torch.no_grad():
        model.eval()
        for images, labels in testloader:
            images = images.cuda()
            labels = labels.cuda()

            # evaluating step
            images = images.view(images.shape[0], -1)
            log_ps = model(images.cuda())
            test_loss += criterion(log_ps, labels.cuda())

            # accuracy step
            ps = torch.exp(log_ps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor))
    test_losses.append(test_loss / len(testloader))
    print(
        "Epoch: {}/{}..".format(e + 1, epochs),
        "Training loss: {:.3f}..".format(running_loss / len(trainloader)),
        "Test loss: {:.3f}..".format(test_loss / len(testloader)),
        "Test Accuracy: {:.3f}".format(accuracy / len(testloader)),
    )