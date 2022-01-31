#date: 2022-01-31T17:05:04Z
#url: https://api.github.com/gists/9a76cb2340a12428f9971934b49c8858
#owner: https://api.github.com/users/ndemir

mean = np.array([0.5, 0.5, 0.5])
std = np.array([0.5, 0.5, 0.5])

train_transformers = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])
val_transformers = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

train_ds = datasets.ImageFolder('./data/hymenoptera_data/train', transform=train_transformers)
val_ds = datasets.ImageFolder('./data/hymenoptera_data/val', transform=val_transformers)
train_ds, val_ds
