#date: 2022-01-31T17:05:09Z
#url: https://api.github.com/gists/d517ffc84afe5bacceb49083381d500d
#owner: https://api.github.com/users/ndemir

model.fc = nn.Linear(model.fc.in_features, 2)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #use CUDA if exists
print('device', device)
model.to(device)
