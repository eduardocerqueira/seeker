#date: 2024-12-17T17:09:59Z
#url: https://api.github.com/gists/a1b97c1665e26d35661a79a9cb7af83d
#owner: https://api.github.com/users/docsallover

from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# Create a DataLoader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)