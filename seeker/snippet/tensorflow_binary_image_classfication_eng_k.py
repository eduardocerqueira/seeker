#date: 2022-06-02T16:51:19Z
#url: https://api.github.com/gists/808aea9c43ba46360d35ad72e67f1327
#owner: https://api.github.com/users/andrea-dagostino

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# we rescale all our images with the rescale parameter
train_datagen = ImageDataGenerator(rescale = 1.0/255)
test_datagen  = ImageDataGenerator(rescale = 1.0/255)

# we use flow_from_directory to create a generator for training
train_generator = train_datagen.flow_from_directory(train_dir,
                                                    batch_size=20,
                                                    class_mode='binary',
                                                    target_size=(150, 150))

# we use flow_from_directory to create a generator for validation
validation_generator =  test_datagen.flow_from_directory(validation_dir,
                                                         batch_size=20,
                                                         class_mode='binary',
                                                         target_size=(150, 150))