#date: 2022-04-21T17:12:58Z
#url: https://api.github.com/gists/9df9d165cbd6252df4c3736f5dd71639
#owner: https://api.github.com/users/diegounzueta

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.keras.applications import ResNet50
from tensorflow import keras

model = Sequential()
model.add(ResNet50(include_top = False, pooling = RESNET50_POOLING_AVERAGE, weights='imagenet'))
model.add(Dense(NUM_CLASSES, activation = DENSE_LAYER_ACTIVATION))

#fix all model weights EXCEPT FOR THE FINAL LAYER (the one we want to train)
for i in model.layers[:-1]:
    i.trainable = False

#compile the model with your favourite optimizer, loss function and metrics 
model.compile(optimizer = "adam", loss = "categorical_crossentropy", metrics = "accuracy")