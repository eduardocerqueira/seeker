#date: 2025-01-03T17:11:24Z
#url: https://api.github.com/gists/130098c64cf7917232051e4ef12612a3
#owner: https://api.github.com/users/PieroPaialungaAI

plt.figure(figsize=(20,5))
plt.subplot(1,2,1)
plt.scatter(X_PCA[:,1],y,s=20)
plt.xlabel('Component PCA 2')
plt.ylabel('Y (target)')
plt.subplot(1,2,2)
plt.scatter(X_PCA[:,0],y,s=20)
plt.xlabel('Component PCA 1')
plt.ylabel('Y (target)')