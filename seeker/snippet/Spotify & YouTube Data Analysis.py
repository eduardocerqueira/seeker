#date: 2025-07-14T16:58:25Z
#url: https://api.github.com/gists/d321eb1ed4bdde00f2d697784a843f6f
#owner: https://api.github.com/users/aniya-mitchell

#a
import numpy as np
import matplotlib.pyplot as plt

class NeuralNetwork():
    def __init__(self, learning_r):
        self.weights = np.random.randn(3, 1)
        self.learning_rate = learning_r
        self.history = []

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def forward_propagation(self, inputs):
        return self.sigmoid(np.dot(inputs, self.weights))

    def train(self, inputs_train, labels_train, num_train_iterations):
        m = inputs_train.shape[0]  # Number of training examples
        for epoch in range(num_train_iterations):
            predictions = self.forward_propagation(inputs_train)

            # Compute error
            error = predictions - labels_train
            cost = np.mean(error ** 2)  # Mean Squared Error

            # Compute gradient
            gradient = np.dot(inputs_train.T, error) / m

            # Update weights
            self.weights -= self.learning_rate * gradient

            # Store weights and cost
            self.history.append((self.weights.copy(), cost))

#b
inputs = np.array([[1, 1], [1, 0], [0, 1], [0.5, -1], [0.5, 3], [0.7, 2], [-1, 0], [-1, 1], [2, 0], [0, 0]])
labels = np.array([[1], [1], [0], [0], [1], [1], [0], [0], [1], [0]])

# Add bias
inputs_bias = np.hstack((inputs, np.ones((inputs.shape[0], 1))))

#c, d, e
# Train and visualize
learning_rates = [1, 0.5, 0.1, 0.01]
plt.figure(figsize=(12, 8))
# Plot learning curves
plt.figure(figsize=(12, 8))
for lr in learning_rates:
    nn = NeuralNetwork(learning_r=lr)
    nn.train(inputs_bias, labels, 50)

    costs = [cost for _, cost in nn.history]
    plt.plot(range(50), costs, label=f'LR={lr}')

plt.xlabel('Epochs')
plt.ylabel('Training Cost')
plt.title('Learning Curves for Different Learning Rates')
plt.legend()
plt.show()

# Plot all decision boundaries in a single graph for comparisons
plt.figure(figsize=(8, 6))
for lr in learning_rates:
    nn = NeuralNetwork(learning_r=lr)
    nn.train(inputs_bias, labels, 50)

    x_values = np.linspace(-2, 3, 100)
    y_values = -(nn.weights[0] * x_values + nn.weights[2]) / nn.weights[1]

    plt.plot(x_values, y_values.flatten(), label=f'LR={lr}')

# Plot data points
for j in range(len(labels)):
    plt.scatter(inputs[j, 0], inputs[j, 1],
                color='red' if labels[j] == 0 else 'green',
                marker='s' if labels[j] == 0 else '^',
                edgecolors='k')

plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Decision Boundaries for Different Learning Rates')
plt.legend()
plt.show()