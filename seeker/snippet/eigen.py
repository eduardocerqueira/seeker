#date: 2024-01-17T16:49:01Z
#url: https://api.github.com/gists/6f085bbf3a81aae4d811a24c059d12b1
#owner: https://api.github.com/users/pratapsdev11

def eigen_val_vector(x,k):
  eig_vals,eig_vecs = np.linalg.eig(covariance(x))
  sort_idx=np.argsort(eig_vals)[::-1]
  eig_vals=eig_vals[sort_idx]
  eig_vecs=eig_vecs[:,sort_idx]
  components=eig_vecs[:k]
  explained_variance=np.sum(eig_vals[:k])/np.sum(eig_vals)
  return explained_variance
     