#date: 2022-01-31T17:04:19Z
#url: https://api.github.com/gists/758d25d4b1bc3601e0984ed4a8f84cd3
#owner: https://api.github.com/users/ndemir

model = torch.nn.Linear(2, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)
lrs = []

for epoch in range(10):
    optimizer.step()
    lrs.append(optimizer.param_groups[0]["lr"])
    scheduler.step()

plt.plot(range(10),lrs)
