#date: 2025-08-13T17:07:56Z
#url: https://api.github.com/gists/d9e5be1bb2413e35865d8c38914de51b
#owner: https://api.github.com/users/francesco-s

import numpy as np
import matplotlib.pyplot as plt

class LogisticRegression:
    def __init__(self, learning_rate=0.01, max_iterations=1000, tolerance=1e-6):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.weights = None
        self.bias = None
        self.cost_history = []
        
    def _sigmoid(self, z):
        """Sigmoid activation function with numerical stability"""
        # Clip z to prevent overflow
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    def _compute_cost(self, y_true, y_pred):
        """Compute logistic regression cost (cross-entropy)"""
        # Add small epsilon to prevent log(0)
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        
        cost = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return cost
    
    def fit(self, X, y):
        """Fit logistic regression model using gradient descent"""
        X = np.array(X)
        y = np.array(y)
        
        m, n = X.shape
        
        # Initialize parameters
        self.weights = np.random.normal(0, 0.01, n)
        self.bias = 0
        
        # Gradient descent
        for iteration in range(self.max_iterations):
            # Forward pass
            z = np.dot(X, self.weights) + self.bias
            predictions = self._sigmoid(z)
            
            # Compute cost
            cost = self._compute_cost(y, predictions)
            self.cost_history.append(cost)
            
            # Compute gradients
            error = predictions - y
            dw = np.dot(X.T, error) / m
            db = np.sum(error) / m
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            # Check for convergence
            if iteration > 0 and abs(self.cost_history[-2] - self.cost_history[-1]) < self.tolerance:
                print(f"Converged after {iteration + 1} iterations")
                break
        
        return self
    
    def predict_proba(self, X):
        """Predict class probabilities"""
        X = np.array(X)
        z = np.dot(X, self.weights) + self.bias
        return self._sigmoid(z)
    
    def predict(self, X, threshold=0.5):
        """Make binary predictions"""
        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int)
    
    def score(self, X, y):
        """Calculate accuracy score"""
        predictions = self.predict(X)
        return np.mean(predictions == y)
    
    def plot_cost_history(self):
        """Plot cost function over iterations"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.cost_history)
        plt.title('Cost Function Over Iterations')
        plt.xlabel('Iteration')
        plt.ylabel('Cross-Entropy Cost')
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def plot_decision_boundary(self, X, y, title="Decision Boundary"):
        """Plot decision boundary for 2D data"""
        if X.shape[1] != 2:
            print("Decision boundary plotting only available for 2D data")
            return
            
        plt.figure(figsize=(10, 8))
        
        # Create a mesh
        h = 0.01
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
        
        # Make predictions on the mesh
        mesh_points = np.c_[xx.ravel(), yy.ravel()]
        Z = self.predict_proba(mesh_points)
        Z = Z.reshape(xx.shape)
        
        # Plot the contour and data points
        plt.contourf(xx, yy, Z, levels=50, alpha=0.6, cmap='RdBu')
        plt.colorbar(label='Probability')
        
        # Plot the decision boundary (P = 0.5)
        plt.contour(xx, yy, Z, levels=[0.5], colors='black', linestyles='--', linewidths=2)
        
        # Plot the data points
        scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap='RdBu', edgecolors='black')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title(title)
        plt.show()

# Demonstration with synthetic data
np.random.seed(42)

# Generate linearly separable data
from sklearn.datasets import make_classification
X, y = make_classification(n_samples=1000, n_features=2, n_redundant=0, 
                          n_informative=2, n_clusters_per_class=1, random_state=42)

# Split data
train_size = int(0.8 * len(X))
indices = np.random.permutation(len(X))
train_idx, test_idx = indices[:train_size], indices[train_size:]

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

# Train the model
model = LogisticRegression(learning_rate=0.1, max_iterations=1000)
model.fit(X_train, y_train)

# Evaluate the model
train_accuracy = model.score(X_train, y_train)
test_accuracy = model.score(X_test, y_test)

print(f"Training Accuracy: {train_accuracy:.3f}")
print(f"Test Accuracy: {test_accuracy:.3f}")
print(f"Learned weights: {model.weights}")
print(f"Learned bias: {model.bias:.4f}")

# Plot cost history
model.plot_cost_history()

# Plot decision boundary
model.plot_decision_boundary(X_test, y_test, "Logistic Regression Decision Boundary")

# Compare with scikit-learn
from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression
from sklearn.metrics import accuracy_score

sklearn_model = SklearnLogisticRegression(random_state=42, max_iter=1000)
sklearn_model.fit(X_train, y_train)
sklearn_pred = sklearn_model.predict(X_test)
sklearn_accuracy = accuracy_score(y_test, sklearn_pred)

print(f"\nComparison with Scikit-learn:")
print(f"Our Implementation: {test_accuracy:.3f}")
print(f"Scikit-learn: {sklearn_accuracy:.3f}")
print(f"Scikit-learn weights: {sklearn_model.coef_[0]}")
print(f"Scikit-learn bias: {sklearn_model.intercept_[0]:.4f}")

# Demonstrate sigmoid function
z_range = np.linspace(-10, 10, 100)
sigmoid_values = model._sigmoid(z_range)

plt.figure(figsize=(10, 6))
plt.plot(z_range, sigmoid_values, 'b-', linewidth=2, label='Sigmoid Function')
plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.7, label='Decision Threshold')
plt.axvline(x=0, color='g', linestyle='--', alpha=0.7, label='z = 0')
plt.xlabel('z (Linear Combination)')
plt.ylabel('σ(z)')
plt.title('Sigmoid Activation Function')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Demonstrate probability predictions
sample_predictions = model.predict_proba(X_test[:10])
binary_predictions = model.predict(X_test[:10])

print(f"\nSample Predictions (first 10 test samples):")
print("Probabilities | Binary | Actual")
print("-" * 35)
for i in range(10):
    print(f"{sample_predictions[i]:.3f}     |   {binary_predictions[i]}    |   {y_test[i]}")