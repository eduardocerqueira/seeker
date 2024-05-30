#date: 2024-05-30T17:05:58Z
#url: https://api.github.com/gists/fda886c5c62930087c1f8678d2ebf06f
#owner: https://api.github.com/users/thomasahle

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import tqdm

# Generate random dataset
np.random.seed(0)
x = np.linspace(-1, 1, 300)
y = np.sin(20 * x) + 2 * (x + x**2) + np.random.normal(0, 0.2, x.shape)
x_tensor = torch.tensor(x, dtype=torch.float32).view(-1, 1)
y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)


class GatedSineLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GatedSineLayer, self).__init__()
        self.fc1 = nn.Linear(in_features, out_features)
        self.fc2 = nn.Linear(in_features, out_features)

    def forward(self, x):
        return torch.sin(self.fc1(x)) * self.fc2(x)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = GatedSineLayer(1, 100)
        self.fc2 = GatedSineLayer(100, 100)
        self.fc3 = GatedSineLayer(100, 100)
        self.fc4 = GatedSineLayer(100, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        return x


# Instantiate the network, define the loss function and the optimizer
net = Net()
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=0.01)

# Training parameters
epochs = 500
interval = 10  # Save the model output every `interval` epochs

# Prepare the figure for plotting
fig, ax = plt.subplots()
fig.patch.set_facecolor("black")
ax.set_facecolor("black")
ax.spines["bottom"].set_color("white")
ax.spines["top"].set_color("white")
ax.spines["left"].set_color("white")
ax.spines["right"].set_color("white")
ax.tick_params(axis="x", colors="white")
ax.tick_params(axis="y", colors="white")
ax.yaxis.label.set_color("white")
ax.xaxis.label.set_color("white")
ax.title.set_color("white")

# Prepare the figure for plotting
ax.plot(x, y, "r.")
ax.set_ylim(-2, 3)

# Training loop
frames = []

for epoch in tqdm.tqdm(range(epochs)):
    net.train()
    optimizer.zero_grad()
    outputs = net(x_tensor)
    loss = criterion(outputs, y_tensor)
    loss.backward()
    optimizer.step()

    if epoch % interval == 0:
        net.eval()
        with torch.no_grad():
            pred_y = net(x_tensor).numpy()
            frame = ax.plot(x, pred_y, "y-")
            title = ax.text(
                0.5,
                1.05,
                f"Epoch {epoch}, Loss: {loss.item():.4f}",
                size=plt.rcParams["axes.titlesize"],
                ha="center",
                transform=ax.transAxes,
                color="white",
            )
            frames.append(frame + [title])

ani = animation.ArtistAnimation(fig, frames, interval=50, blit=True)
ani.save("fit_animation.mp4", writer="ffmpeg")
