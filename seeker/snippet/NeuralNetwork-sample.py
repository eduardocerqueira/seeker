#date: 2021-11-24T17:10:20Z
#url: https://api.github.com/gists/7cf7917582d8531148e5efeb0fce31d1
#owner: https://api.github.com/users/SDolha

import numpy as np
np.random.seed(0)

def sigmoid(t):
    return 1 / (1 + np.exp(-t))
def sigmoid_derivative(p):
    return p * (1 - p)

class NeuralNetwork:
    def __init__(self, x, y, weights=2**4, training=2**16):
        self.input = x
        self.weights1 = np.random.rand(self.input.shape[1], weights)
        self.weights2 = np.random.rand(weights, 1)
        self.y = y
        self.output = np.zeros(y.shape)
        for _ in range(training):
            self.train()
    def feedforward(self, x):
        self.layer1 = sigmoid(np.dot(x, self.weights1))
        self.layer2 = sigmoid(np.dot(self.layer1, self.weights2))
        return self.layer2
    def backprop(self):
        d_weights2 = np.dot(self.layer1.T, 2 * (self.y - self.output) * sigmoid_derivative(self.output))
        d_weights1 = np.dot(self.input.T, np.dot(2 * (self.y - self.output) * sigmoid_derivative(self.output), self.weights2.T) * sigmoid_derivative(self.layer1))
        self.weights1 += d_weights1
        self.weights2 += d_weights2
    def train(self):
        self.output = self.feedforward(self.input)
        self.backprop()
    def solve(self, x):
        return np.round(self.feedforward(x))

print("Training:")
x = np.array(([0,0,0], [0,1,0], [0,1,1], [1,0,0], [1,0,1], [1,1,0]))
y = np.array(([0], [0], [0], [1], [1], [1]))
print(x)
print(y)

print("Configuration:")
NN = NeuralNetwork(x, y)
print(NN.weights1)
print(NN.weights2)

print("Input:")
I = np.array(([0,0,1], [1,1,1]))
print(I)

print("Prediction:")
o = NN.solve(I)
print(o)
