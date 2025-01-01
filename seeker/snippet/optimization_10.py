#date: 2025-01-01T16:22:43Z
#url: https://api.github.com/gists/effe9be465e4fa0cb8dfa4e716ed2e02
#owner: https://api.github.com/users/PieroPaialungaAI

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

X_train, X_test, y_train, y_test = X[train_set], X[test_set], Y[train_set], Y[test_set]
kernel = 1 * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
gaussian_process = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)
gaussian_process.fit(X_train.reshape(-1,1), y_train)
mean, std = gaussian_process.predict(X.reshape(-1,1),return_std=True)
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
plt.show()