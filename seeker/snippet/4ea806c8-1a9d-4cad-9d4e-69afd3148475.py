#date: 2021-12-13T16:54:46Z
#url: https://api.github.com/gists/dfedc202124b09cf36cbd07131d3826b
#owner: https://api.github.com/users/ifakat

# Reshaping X_train for efficient modelling
X_train = np.reshape(X_train, (X_train.shape[0],X_train.shape[1],1))
