#date: 2025-08-13T17:06:24Z
#url: https://api.github.com/gists/45599ee098592166f55e7e0db7ff14df
#owner: https://api.github.com/users/francesco-s

import numpy as np
import matplotlib.pyplot as plt

class LinearRegression:
    def __init__(self, learning_rate=0.01, max_iterations=1000, tolerance=1e-6, method='gradient_descent'):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.method = method
        self.weights = None
        self.bias = None
        self.cost_history = []
        
    def _add_bias(self, X):
        """Add bias column to feature matrix"""
        return np.hstack([np.ones((X.shape[0], 1)), X])
    
    def _compute_cost(self, X, y):
        """Compute mean squared error cost"""
        m = X.shape[0]
        predictions = self.predict(X)
        cost = np.sum((predictions - y) ** 2) / (2 * m)
        return cost
    
    def fit(self, X, y):
        """Fit linear regression model using specified method"""
        X = np.array(X)
        y = np.array(y)
        
        if self.method == 'normal_equation':
            self._fit_normal_equation(X, y)
        else:
            self._fit_gradient_descent(X, y)
        
        return self
    
    def _fit_normal_equation(self, X, y):
        """Fit using normal equation: θ = (X^T X)^(-1) X^T y"""
        X_with_bias = self._add_bias(X)
        
        # Normal equation: θ = (X^T X)^(-1) X^T y
        XTX = np.dot(X_with_bias.T, X_with_bias)
        XTy = np.dot(X_with_bias.T, y)
        
        # Add regularization for numerical stability
        regularization = 1e-8 * np.eye(XTX.shape[0])
        theta = np.linalg.solve(XTX + regularization, XTy)
        
        self.bias = theta[0]
        self.weights = theta[1:]
    
    def _fit_gradient_descent(self, X, y):
        """Fit using gradient descent"""
        m, n = X.shape
        
        # Initialize parameters
        self.weights = np.random.normal(0, 0.01, n)
        self.bias = 0
        
        # Gradient descent
        for iteration in range(self.max_iterations):
            # Forward pass
            predictions = self.predict(X)
            
            # Compute cost
            cost = np.sum((predictions - y) ** 2) / (2 * m)
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
    
    def predict(self, X):
        """Make predictions"""
        X = np.array(X)
        return np.dot(X, self.weights) + self.bias
    
    def score(self, X, y):
        """Calculate R² score"""
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot)
    
    def plot_cost_history(self):
        """Plot cost function over iterations (only for gradient descent)"""
        if len(self.cost_history) > 0:
            plt.figure(figsize=(10, 6))
            plt.plot(self.cost_history)
            plt.title('Cost Function Over Iterations')
            plt.xlabel('Iteration')
            plt.ylabel('Mean Squared Error')
            plt.grid(True, alpha=0.3)
            plt.show()
        else:
            print("No cost history available (using normal equation)")

# Demonstration with synthetic data
np.random.seed(42)

# Generate sample data
X = np.random.randn(100, 2)  # 2 features
true_weights = np.array([3, -2])
true_bias = 1
y = np.dot(X, true_weights) + true_bias + np.random.normal(0, 0.5, 100)

# Split data
train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Compare both methods
methods = ['gradient_descent', 'normal_equation']
models = {}

for method in methods:
    print(f"\n=== {method.replace('_', ' ').title()} ===")
    
    # Train model
    model = LinearRegression(method=method, learning_rate=0.1, max_iterations=1000)
    model.fit(X_train, y_train)
    models[method] = model
    
    # Evaluate
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    
    print(f"Learned weights: {model.weights}")
    print(f"Learned bias: {model.bias:.4f}")
    print(f"True weights: {true_weights}")
    print(f"True bias: {true_bias}")
    print(f"Train R²: {train_score:.4f}")
    print(f"Test R²: {test_score:.4f}")

# Plot cost history for gradient descent
models['gradient_descent'].plot_cost_history()

# Visualize results for 1D case
np.random.seed(42)
X_1d = np.random.randn(100, 1)
y_1d = 2 + 3 * X_1d.flatten() + np.random.randn(100) * 0.5

# Train models
gd_model = LinearRegression(method='gradient_descent', max_iterations=1000)
ne_model = LinearRegression(method='normal_equation')

gd_model.fit(X_1d, y_1d)
ne_model.fit(X_1d, y_1d)

# Plot results
plt.figure(figsize=(15, 5))

# Gradient Descent
plt.subplot(1, 3, 1)
plt.scatter(X_1d, y_1d, alpha=0.6, s=20)
X_plot = np.linspace(X_1d.min(), X_1d.max(), 100).reshape(-1, 1)
y_plot = gd_model.predict(X_plot)
plt.plot(X_plot, y_plot, 'r-', linewidth=2, label=f'GD: y = {gd_model.bias:.2f} + {gd_model.weights[0]:.2f}x')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Gradient Descent')
plt.legend()
plt.grid(True, alpha=0.3)

# Normal Equation
plt.subplot(1, 3, 2)
plt.scatter(X_1d, y_1d, alpha=0.6, s=20)
y_plot = ne_model.predict(X_plot)
plt.plot(X_plot, y_plot, 'g-', linewidth=2, label=f'NE: y = {ne_model.bias:.2f} + {ne_model.weights[0]:.2f}x')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Normal Equation')
plt.legend()
plt.grid(True, alpha=0.3)

# Comparison
plt.subplot(1, 3, 3)
plt.scatter(X_1d, y_1d, alpha=0.6, s=20, label='Data')
plt.plot(X_plot, gd_model.predict(X_plot), 'r-', linewidth=2, label='Gradient Descent')
plt.plot(X_plot, ne_model.predict(X_plot), 'g--', linewidth=2, label='Normal Equation')
plt.plot(X_plot, 2 + 3 * X_plot, 'k:', linewidth=2, label='True Function')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Comparison')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"\nGradient Descent - Weight: {gd_model.weights[0]:.4f}, Bias: {gd_model.bias:.4f}")
print(f"Normal Equation - Weight: {ne_model.weights[0]:.4f}, Bias: {ne_model.bias:.4f}")
print(f"True Parameters - Weight: 3.0000, Bias: 2.0000")