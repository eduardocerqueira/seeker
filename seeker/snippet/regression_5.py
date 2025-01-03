#date: 2025-01-03T17:10:20Z
#url: https://api.github.com/gists/777ac6ff302d62e368608b0168a847ed
#owner: https://api.github.com/users/PieroPaialungaAI

noise = np.random.normal(0,0.1,X_1.shape)
y = 5*X_PCA[:,1].reshape(-1,1)+noise
y = y[:,0]
plt.figure(figsize=(20,5))
plt.subplot(1,2,1)
plt.scatter(X[:,1],y,s=20)
plt.xlabel(r'$X_2$')
plt.ylabel('Y (target)')
plt.subplot(1,2,2)
plt.scatter(X[:,0],y,s=20)
plt.xlabel(r'$X_1$')
plt.ylabel('Y (target)')