#date: 2022-01-31T17:05:08Z
#url: https://api.github.com/gists/bc833ef226509203d6aff62812251372
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
