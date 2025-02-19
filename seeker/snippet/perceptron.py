#date: 2025-02-19T17:01:45Z
#url: https://api.github.com/gists/e002f8f20a0a41c1af7a1c29bdb8e86e
#owner: https://api.github.com/users/ranielcsar

import numpy as np


class Perceptron:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.bias = None
        self.weights = None

    def activation_fn(self, net_input):
        return np.where(net_input > 0, 1, 0)

    def fit(self, features, targets):
        _, n_features = features.shape

        self.weights = np.random.uniform(size=n_features, low=-0.5, high=0.5)
        self.bias = np.random.uniform(low=-0.5, high=0.5)

        for _ in range(self.n_iterations):
            for example_index, example_features in enumerate(features):
                net_input = np.dot(example_features, self.weights) + self.bias
                y_predicted = self.activation_fn(net_input)
                self._update_weights(
                    example_features, targets[example_index], y_predicted
                )

    def _update_weights(self, example_features, y_actual, y_predicted):
        error = y_actual - y_predicted
        weight_correction = self.learning_rate * error
        self.weights = self.weights + weight_correction * example_features
        self.bias = self.bias + weight_correction

    def predict(self, features):
        net_input = np.dot(features, self.weights) + self.bias
        y_predicted = self.activation_fn(net_input)
        return y_predicted
