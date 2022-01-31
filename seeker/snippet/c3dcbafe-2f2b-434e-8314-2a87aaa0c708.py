#date: 2022-01-31T17:04:15Z
#url: https://api.github.com/gists/4de031149da9cda6499931746a1aefd6
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
