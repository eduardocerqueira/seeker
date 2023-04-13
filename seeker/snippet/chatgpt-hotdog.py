#date: 2023-04-13T16:56:19Z
#url: https://api.github.com/gists/d97d91883e0439742df7d3af38ca06f2
#owner: https://api.github.com/users/gogococo

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
import cv2

# Configuration
img_width, img_height = 224, 224
input_shape = (img_width, img_height, 3)
batch_size = 32
no_epochs = 20
no_classes = 1
validation_split = 0.2
verbosity = 1

# Load data using ImageDataGenerator
datagen = ImageDataGenerator(rescale=1./255, validation_split=validation_split)
train_generator = datagen.flow_from_directory(
    'hotdog/train/',
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary',
    subset='training')
val_generator = datagen.flow_from_directory(
    'hotdog/train/',
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary',
    subset='validation')

# Model creation
def create_model():
  model = Sequential()
  model.add(Conv2D(16, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Flatten())
  model.add(Dense(256, activation='relu'))
  model.add(Dropout(0.5))
  model.add(Dense(1, activation='sigmoid'))
  return model

# Model compilation
def compile_model(model):
  model.compile(loss='binary_crossentropy',
              optimizer=tf.keras.optimizers.Adam(),
              metrics=['accuracy'])
  return model

# Model training
def train_model(model, train_generator, val_generator):
  model.fit(train_generator,
            epochs=no_epochs,
            verbose=verbosity,
            validation_data=val_generator)
  return model

# Model testing
def test_model(model, test_generator):
  score = model.evaluate(test_generator, verbose=0)
  print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')
  return model

# Predict
def predict_model(model, image_path):
    
  # Load and preprocess image
    img = cv2.imread(image_path)
    resized_img = cv2.resize(img, (img_width, img_height))
    input_img = np.array(resized_img).reshape(1, img_width, img_height, 3)/255.

    # Predict class probabilities
    class_probabilities = model.predict(input_img)

    # Check if image contains a hotdog
    contains_hotdog = class_probabilities[0, 0] > 0.5
    print(f'The image {image_path} contains a hotdog: {contains_hotdog}')

# Create and train the model
model = create_model()
model = compile_model(model)
model = train_model(model, train_generator, val_generator)

# Load test data using ImageDataGenerator
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory

predict_model(model, 'classic-hot-dog.png')
predict_model(model, 'person.png')
predict_model(model, 'burger.jpg')