#date: 2025-01-01T16:29:29Z
#url: https://api.github.com/gists/7c92b293a59638b6b4864662518484f7
#owner: https://api.github.com/users/PieroPaialungaAI

import matplotlib.pyplot as plt
x_values = np.linspace(0, 10, 400)
y_values = np.linspace(0, 10, 400)
# Create a meshgrid
X, Y = np.meshgrid(x_values, y_values)
# Compute the function values on the grid
Z = func_to_minimize(X, Y)
# Plot the contour
plt.figure(figsize=(8, 6))
contour = plt.contourf(X, Y, Z, levels=50, cmap='viridis')
plt.colorbar(contour)
plt.title('Contour Plot of $f(x, y) = x \\sin(4x) + 1.1y \\sin(2y)$')
plt.xlabel('x')
plt.ylabel('y')
plt.plot(np.array(ga_instance.solutions)[:,0],np.array(ga_instance.solutions)[:,1],'x',markersize=2,color='firebrick')
plt.xlim(0,10)
plt.ylim(0,10)
plt.show()