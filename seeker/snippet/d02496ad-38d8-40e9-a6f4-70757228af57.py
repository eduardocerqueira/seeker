#date: 2022-01-31T17:04:17Z
#url: https://api.github.com/gists/57c7c6113681f479bc904f767a266345
#owner: https://api.github.com/users/ndemir

batch_size=64
train_loader = DataLoader(train_ds, batch_size, shuffle=True)
val_loader = DataLoader(val_ds, batch_size)
