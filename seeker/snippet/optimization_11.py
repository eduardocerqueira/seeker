#date: 2025-01-01T16:24:06Z
#url: https://api.github.com/gists/025d3946647aad2fdda131675cf894a5
#owner: https://api.github.com/users/PieroPaialungaAI

from scipy.stats import norm

def expected_improvement(x, gp_model, best_y):
    y_pred, y_std = gp_model.predict(x.reshape(-1, 1), return_std=True)
    z = (best_y - y_pred) / y_std
    ei = (best_y - y_pred) * norm.cdf(z) + y_std * norm.pdf(z)
    return ei

gaussian_process = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)
kernel = 1 * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
gaussian_process.fit(X_train.reshape(-1,1), y_train)
predicted_y = gaussian_process.predict(X.reshape(-1, 1), return_std=False)
best_idx = np.argmin(predicted_y)
best_x = X[best_idx]
best_y = Y[best_idx]
X_range = np.linspace(X.min(), X.max(), 1000)
ei = expected_improvement(X_range, gaussian_process, best_y)

# Plot the expected improvement
plt.figure(figsize=(10, 6))
plt.subplot(2,1,2)
plt.plot(X_range, ei, color='green', label='Expected Improvement')
plt.xlabel('x')
plt.ylabel('Expected Improvement')
plt.title('Expected Improvement')
plt.legend()
plt.grid()
plt.subplot(2,1,1)
plt.plot(X,mean,color='navy')
plt.fill_between(X,mean-2*std,mean+2*std,color='navy',alpha=0.2)
plt.plot(X, Y,label='Target Function',ls='--',color='darkorange',lw=0.8)
plt.plot(X[train_set],Y[train_set],'x',color='firebrick',label='Training Set')
plt.plot(X[test_set],Y[test_set],'x',color='navy',label='Test Set')
plt.xlabel('W')
plt.ylabel('y')
plt.title(r'$y=-\sin(x)+\frac{6}{5}\cos(3x)$',fontsize=12,weight='bold')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()