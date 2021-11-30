#date: 2021-11-30T17:08:55Z
#url: https://api.github.com/gists/4feb0687e071286464c4ec962558b3eb
#owner: https://api.github.com/users/Steboss89

adj_hat = adj +  np.identity(34) # add the self term, so a node is connected with itself
deg_hat = np.array(np.sum(adj_hat, axis=0))[0] 
deg_hat = np.matrix(np.diag(deg_hat))
