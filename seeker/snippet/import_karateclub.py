#date: 2021-11-30T17:01:56Z
#url: https://api.github.com/gists/9766362b2326bea55a6378a9ec035151
#owner: https://api.github.com/users/Steboss89

# Create the Adjacency matrix 
order = np.arange(0,34,1) # 34 nodes
Adj = to_numpy_matrix(karate_club, nodelist=order) 
# Create graph features 
one_hot = OneHotEncoder()
X = np.reshape(order, (-1, 1)) # simple one-hot encoder
X = one_hot.fit_transform(X).toarray()