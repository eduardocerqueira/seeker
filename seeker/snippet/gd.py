#date: 2024-04-15T17:08:14Z
#url: https://api.github.com/gists/3c315158089b30e60e07a48d9e9d7eb9
#owner: https://api.github.com/users/wlinds

import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

np.random.seed(42)
samples = 1000

X = 2 * np.random.rand(samples,1)                   # Features with uniform distribution
epsilon = np.random.normal(0,1, size=(samples,1))   # Random disturbance
y = 2 + 9 * X + epsilon                             # Simple linear regression model
X = np.c_[np.ones(samples), X]                      # Adding a column of ones to X for the intercept term in linear regression

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42) # Split the data

def gradient_descent(X, y, learning_rate=.1, epochs=100): 
    m = len(X)
    theta = np.random.randn(X.shape[1],1)           # Random weights for each feature

    for _ in range(epochs):                         # Compute gradient using the mean squared error derivative
        gradient = 2 / m * X.T @ (X@theta-y)        # Matrix multiplication shorthand
        theta -= learning_rate * gradient           # Update weights using the gradient and learning rate (eta)

    return theta

steps = range(1,200,1)

# Compute theta values for each epoch step and flatten into a 1D array
thetas = np.array([gradient_descent(X_train, y_train, epochs=epoch).reshape(-1) for epoch in steps])

# Plot

plt.figure(figsize=(10, 6))
plt.plot(steps, thetas, label=[r"$\beta_0$", r"$\beta_1$"], linewidth=2)
plt.axhline(y=2, linestyle="--", color="red", label=r"True $\beta_0$", linewidth=2)
plt.axhline(y=9, linestyle="--", color="blue", label=r"True $\beta_1$", linewidth=2)
plt.xlabel("Epochs", fontsize=14)
plt.ylabel(r"$\theta$ values", fontsize=14)
plt.title("Batch Gradient Descent", fontsize=16)
plt.legend(fontsize=12)
plt.grid(True)

plt.show()
fig.savefig(fname="plot_GD.png")