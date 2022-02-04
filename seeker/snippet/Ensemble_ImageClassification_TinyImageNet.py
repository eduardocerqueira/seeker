#date: 2022-02-04T16:53:04Z
#url: https://api.github.com/gists/b3246d8861a74a30c26bf5e1acb5d49c
#owner: https://api.github.com/users/alexppppp

train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False)
num_epochs = 10