#date: 2023-08-30T17:01:06Z
#url: https://api.github.com/gists/ce5521b77b9b9445a54bcd34a775ae80
#owner: https://api.github.com/users/sudo-sh

import numpy as np
import cvxpy
import time

np.random.seed(42)  # for reproducibility

# Generate data dimensions
m, n = 20, 16
x_gt = np.random.choice(np.arange(0,3), size = n)
# Generate synthetic data
H = np.random.randn(m, n)
y = H@x_gt

# y = np.random.randn(m)

# print(H)
# print(y)
# Declare the binary integer-valued optimization variable
x = cvxpy.Variable(n, integer=True)

# # Set of possible values
# P = [2, 4, 8, 9]

# # Create binary variables for each value in P
# z = cvxpy.Variable(len(P), boolean=True)

# # Constraint to ensure only one value is chosen
constraints = [x>=0, x<=3]

# # Assign the value to x based on the chosen value from P
# constraints.append(x == P @ z)



# Set up the L2-norm minimization problem
obj = cvxpy.Minimize(cvxpy.norm(y - H @ x, 2))
prob = cvxpy.Problem(obj, constraints)

# print("Solver status:", prob.status)
start = time.time()
# Solve the problem using an appropriate solver
sol = prob.solve(solver='ECOS_BB')
end = time.time()
print("Time Taken:", end-start)
# Get the optimal value of x
optimal_x = x.value

# Convert binary solution to integer solution
integer_solution = np.round(optimal_x).astype(int)

# Compute the final error
final_error = np.linalg.norm(y - H @ integer_solution, 2)/ np.linalg.norm(y)

# Print the optimal value of x
# print("Optimal x (binary):", optimal_x)

# Print the integer solution of x
print("Optimal x (integer):", integer_solution)
print("Ground Truth:", x_gt)

# Print the final error
print("Final error:", final_error)
