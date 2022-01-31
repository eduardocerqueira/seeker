#date: 2022-01-31T17:04:20Z
#url: https://api.github.com/gists/e9b7adcd1a87579bb9d3892b77b97d6d
#owner: https://api.github.com/users/ndemir

model.fc = nn.Linear(model.fc.in_features, 2)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #use CUDA if exists
print('device', device)
model.to(device)
