#date: 2023-05-05T16:41:08Z
#url: https://api.github.com/gists/10b8cd8c82c0ab735260e04c6efde5a1
#owner: https://api.github.com/users/mohammadnr2817

from numpy import argmax
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.utils import load_img
from keras.utils import img_to_array
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import SGD
import matplotlib.pyplot as plt
import os.path
import sys

# define the model file name
model_file_name = 'mnist_cnn_test_1.h5'


# load train and test dataset
def load_dataset():
	# load dataset
	(trainX, trainY), (testX, testY) = mnist.load_data()
	# reshape dataset to have a single channel
	trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
	testX = testX.reshape((testX.shape[0], 28, 28, 1))
	# one hot encode target values
	trainY = to_categorical(trainY)
	testY = to_categorical(testY)
	return trainX, trainY, testX, testY

# scale pixels
def prep_pixels(train, test):
	# convert from integers to floats
	train_norm = train.astype('float32')
	test_norm = test.astype('float32')
	# normalize to range 0-1
	train_norm = train_norm / 255.0
	test_norm = test_norm / 255.0
	# return normalized images
	return train_norm, test_norm

# define cnn model
def define_model():
	model = Sequential()
	model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
	model.add(MaxPooling2D((2, 2)))
	model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
	model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
	model.add(MaxPooling2D((2, 2)))
	model.add(Flatten())
	model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
	model.add(Dense(10, activation='softmax'))
	# compile model
	opt = SGD(learning_rate=0.01, momentum=0.9)
	model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
	return model

# run the test harness for evaluating a model
def run_test_harness():
	# load dataset
	trainX, trainY, testX, testY = load_dataset()
	# prepare pixel data
	trainX, testX = prep_pixels(trainX, testX)
	# define model
	model = define_model()
	# fit model
	model.fit(trainX, trainY, epochs=10, batch_size=32, verbose=0)
	# save model
	model.save(model_file_name)


# load and prepare the image
def load_image(filename):
	# load the image
	img = load_img(filename, grayscale=True, target_size=(28, 28))
	# convert to array
	img = img_to_array(img)
	# reshape into a single sample with 1 channel
	img = img.reshape(1, 28, 28, 1)
	# prepare pixel data
	img = img.astype('float32')
	img = img / 255.0
	return img

# load an image and predict the class
def run_example(path):
	# load the image
	img = load_image(path)
	# load model
	model = load_model(model_file_name)
	# predict the class
	predict_value = model.predict(img)
	digit = argmax(predict_value)
	print(digit)
	plt.imshow(img[0], cmap='gray')
	plt.title('Predicted label: ' + str(digit))
	plt.show()
	

# -----------------------------------------------
# -----------------------------------------------
# -----------------------------------------------
# ------------------ENTRY POINT------------------
# -----------------------------------------------
# -----------------------------------------------
# -----------------------------------------------


# force data re-training
re_train = input('re train data? (0 -> false | 1 -> true): ')

# end program if input condition not satisfied
if re_train != "0" and re_train != "1" and re_train != "":
  print("input condition not satisfied")
  sys.exit()


if os.path.isfile(model_file_name) and (re_train == "0" or re_train == ""):
  # load model
  model = load_model(model_file_name)
else:
    run_test_harness()


run_example(path='test_2_1.png')