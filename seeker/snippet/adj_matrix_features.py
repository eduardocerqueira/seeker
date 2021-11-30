#date: 2021-11-30T17:02:30Z
#url: https://api.github.com/gists/f19ab9f71d6af85d996c18e7f6b018d6
#owner: https://api.github.com/users/Steboss89

# Create the Adjacency matrix 
order = np.arange(0,34,1) # 34 nodes
Adj = to_numpy_matrix(karate_club, nodelist=order) 
# Create graph features 
one_hot = OneHotEncoder()
X = np.reshape(order, (-1, 1)) # simple one-hot encoder
X = one_hot.fit_transform(X).toarray()