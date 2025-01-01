#date: 2025-01-01T16:26:31Z
#url: https://api.github.com/gists/b47a479c2b0a4b6cb594fc0e93ca4881
#owner: https://api.github.com/users/PieroPaialungaAI

def black_box_function(x):
    return -np.sin(x) + 1.2 * np.cos(3 * x)
num_iterations = 9
plt.figure(figsize=(10, 6))
start_X, start_Y = X_train, y_train
gaussian_process = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)
X_range = np.linspace(X.min(), X.max(), 1000)
plt.figure(figsize=(20,10))
for i in range(num_iterations):
    plt.subplot(3,3,i+1)
    # Fit the Gaussian process model to the sampled points
    gaussian_process.fit(start_X.reshape(-1, 1), start_Y)

    # Determine the point with the highest observed function value
    best_idx = np.argmin(start_Y)
    best_x = start_X[best_idx]
    best_y = start_Y[best_idx]

    # Set the value of beta for the UCB acquisition function

    # Generate the Upper Confidence Bound (UCB) using the Gaussian process model
    computed_ei = expected_improvement(X_range, gaussian_process, best_y)
    y_pred, y_pred_boundaries = gaussian_process.predict(X_range.reshape(-1,1), return_std=True)

    # Plot the UCB function
    plt.fill_between(X_range, y_pred-2*y_pred_boundaries, y_pred+2*y_pred_boundaries, color='navy', label='Uncertainty Boundaries',alpha=0.2)
    # Plot the black box function, surrogate function, previous points, and new points
    plt.plot(X_range, black_box_function(X_range), color='darkorange', label='Black Box Function')
    #plt.plot(x_range, ucb, color='red', linestyle='dashed', label='Surrogate Function')
    plt.scatter(start_X, start_Y, color='red', label='Previous Points')
    plt.plot(X_range, y_pred,color='navy',label='GPR Prediction ')

    new_x = X_range[np.argmax(computed_ei)]  # Select the next point based on UCB
    new_y = black_box_function(new_x)
    start_X = np.append(start_X, new_x)
    start_Y = np.append(start_Y, new_y)
    plt.scatter(new_x, new_y, color='green', label='New Points')

    #plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f"Iteration #{i+1}")
    plt.legend(fontsize=4)