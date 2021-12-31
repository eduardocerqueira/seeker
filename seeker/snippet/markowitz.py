#date: 2021-12-31T16:59:27Z
#url: https://api.github.com/gists/f33c1f98b20224f351e0c9cbb4aa6a33
#owner: https://api.github.com/users/borab96

def random_portfolios(num_samples, mean_returns, cov_matrix, risk_free_rate=0.05):
    vols, rets, objective = np.zeros(num_samples), np.zeros(num_samples), np.zeros(num_samples)
    for i in range(0, num_samples):
        weights = np.random.random(len(mean_returns))
        vols[i] = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        rets[i] =  mean_returns.dot(weights)
        objective[i] = (rets[i]-risk_free_rate)/vols[i]
    return rets, vols, objective
rets, vols, obj = random_portfolios(40000, mean, cov)
plt.scatter(vols,rets,c=obj,cmap='autumn_r', marker='o', s=0.1, alpha=0.2)
