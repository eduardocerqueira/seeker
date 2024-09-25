#date: 2024-09-25T17:00:13Z
#url: https://api.github.com/gists/93e9c9196a84fee4a7a96f2048b6d1fe
#owner: https://api.github.com/users/MaximeVandegar

class Dataset():

    def __init__(self, data_path):
        self.data_path = data_path
        self.files = [f for f in listdir(data_path) if isfile(join(data_path, f))]
        self.len = len(self.files)

    def __len__(self):
        return self.len

    def __getitem__(self, index):

        transform_list = []
        transform_list += [transforms.Resize(64)]
        transform_list += [transforms.CenterCrop(64)]
        transform_list += [transforms.ToTensor()]
        transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        transform = transforms.Compose(transform_list)
        return transform(Image.open(f'{self.data_path}/' + self.files[index]).convert('RGB'))