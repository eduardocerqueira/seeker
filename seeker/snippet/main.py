#date: 2022-09-15T17:07:01Z
#url: https://api.github.com/gists/c74919dafeb0e0c7dc6c2c6416086c3c
#owner: https://api.github.com/users/ai22m019

# This is a sample Python script.

# Press Umschalt+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

from keras.datasets import mnist

(train_X, train_y), (test_X, test_y) = mnist.load_data()

print('X_train: ' + str(train_X.shape))
print('Y_train: ' + str(train_y.shape))
print('X_test:  '  + str(test_X.shape))
print('Y_test:  '  + str(test_y.shape))

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
