#date: 2022-01-31T17:05:06Z
#url: https://api.github.com/gists/3bc203d1ff83b1b562b7679259fbee98
#owner: https://api.github.com/users/ndemir

batch_size=64
train_loader = DataLoader(train_ds, batch_size, shuffle=True)
val_loader = DataLoader(val_ds, batch_size)
