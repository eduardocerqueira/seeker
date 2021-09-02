#date: 2021-09-02T16:51:16Z
#url: https://api.github.com/gists/348f55c1ef5f09b122fb961812b22d06
#owner: https://api.github.com/users/aswinvk28

def print_images_in_grid(rows, cols, filenames, dataset='train'):
    train_images = []
    train_classes = []
    index = 0
    function = load_train_dataset if dataset == 'train' else load_test_dataset
    for image, string in function(filenames):
        if len(train_images) == 0:
            train_images = [image.numpy()]
        else:
            train_images = train_images + [image.numpy()]
        train_classes = train_classes + \
        [string.numpy().decode() if dataset == 'test' else CLASSES[string.numpy()]]
        index += 1
        if index >= rows*cols:
            break
    plt.figure(figsize=(16, (rows/cols)*12))
    for i in range(rows*cols):
        plt.subplot(rows, cols, i+1)
        plt.imshow(train_images[i])
        plt.title(train_classes[i])
    plt.show()