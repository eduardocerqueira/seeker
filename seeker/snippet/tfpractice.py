#date: 2021-11-15T16:59:33Z
#url: https://api.github.com/gists/f88123a04daba883c3aaa68a825276e1
#owner: https://api.github.com/users/moggs2

import tensorflow as tf
import numpy as np
import termplotlib as tpl
import plotext as plt
x1dimension=np.array([-8.0, -3.0, -1.5, 4.0, 6.0, 7.0, 18.0, 12.0])
x = np.array([[-7.0], [-3.0], [-9.0], [1.0], [7.0], [7.0], [10.0], [13.0]])
y = np.array([3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0, 24.0])

print(x.shape)

# X und Y sind Daten und Label fuer die Regression

# Creating a model from layers https://www.tensorflow.org/api_docs/python/tf/keras/Sequential

inputs = tf.keras.Input(shape=[1,])
layer=tf.keras.layers.Dense(1)(inputs)
layer2=tf.keras.layers.Dense(1)(layer)
layer3=tf.keras.layers.Dense(1)(layer2)
outputs = tf.keras.layers.Dense(1)(layer3)

model = tf.keras.Model(inputs=inputs,outputs=outputs)

model.compile(loss=tf.keras.losses.mae, # mae is short for mean absolute error
              optimizer=tf.keras.optimizers.SGD(), # SGD is short for stochastic gradient descent
              metrics=["mae"])
              
model.fit(x, y, epochs=100)

plt.clp()
plt.scatter(x1dimension, y, "regressinplot")
plt.show()


print(model.predict([17.0]))


import pandas as pd

insurance = pd.read_csv("https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv")

print(insurance)
insurance.sample(frac=1).reset_index(drop=True)   #shuffling the whole dataframe

insurance_one_hot = pd.get_dummies(insurance)

print(insurance_one_hot)

x = insurance_one_hot.drop("charges", axis=1)
y = insurance_one_hot["charges"]

print(str(len(x)) + " " + str(len(y)))


traininglen=round(len(x)*0.8, 0)
print(traininglen)
testlen=len(x)-traininglen
print(testlen)

x_train=x[:int(traininglen)]
x_test=x[int(traininglen):]
print(len(x_train))
print(len(x_test))

print(x_train[0:1])


y_train=y[:int(traininglen)]
y_test=y[int(traininglen):]
print(y_train[0:1])

#housingdata=tf.keras.datasets.boston_housing.load_data(path='boston_housing.npz', test_split=0.2, seed=113)

#x_train=housingdata[0][0]
#y_train=housingdata[0][1]
#x_test=housingdata[1][0]
#y_test=housingdata[1][1]

#print(x_test)
#print(y_test)

inputs = tf.keras.Input(shape=[11,])
layer=tf.keras.layers.Dense(400)(inputs)
layer2=tf.keras.layers.Dense(10)(layer)
outputs = tf.keras.layers.Dense(1)(layer2)

model = tf.keras.Model(inputs=inputs,outputs=outputs)

model.compile(loss=tf.keras.losses.mae, # mae is short for mean absolute error
              optimizer=tf.keras.optimizers.Adam(),
              #optimizer=tf.keras.optimizers.SGD(), # SGD is short for stochastic gradient descent
              metrics=["mae"])
              
fittingdiagram=model.fit(x_train, y_train, epochs=100, verbose=0)
print(fittingdiagram.history)
#tpl.figure(fittingdiagram.history['loss']).show()
plt.clp()
plt.plot(fittingdiagram.history['loss'], label="loss")
plt.plot(fittingdiagram.history['mae'], label="mae")
plt.show()

model.evaluate(x_test, y_test)

prediction=model.predict(x_test)
print("Prediction: ")
print(prediction)
