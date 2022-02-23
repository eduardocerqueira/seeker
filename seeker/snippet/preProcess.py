#date: 2022-02-23T16:56:30Z
#url: https://api.github.com/gists/3f32a920b70531093cce366bfcac6e1c
#owner: https://api.github.com/users/kayathiri4

def add_noise(img):
    '''Add random noise to an image'''
    VARIABILITY = 50
    deviation = VARIABILITY*random.random()
    noise = np.random.normal(0, deviation, img.shape)
    img += noise
    np.clip(img, 0., 255.)
    return img

# DataGenerator to add noise
datagen = ImageDataGenerator(preprocessing_function=add_noise)
