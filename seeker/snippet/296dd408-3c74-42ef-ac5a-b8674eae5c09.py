#date: 2021-12-13T16:54:46Z
#url: https://api.github.com/gists/d18dd701efa18c359588092e75696458
#owner: https://api.github.com/users/ifakat

# Since LSTMs store long term memory state, we create a data structure with 60 timesteps and 1 output
# So for each element of training set, we have 60 previous training set elements 
X_train = []
y_train = []
for i in range(60,2769):
    X_train.append(training_set_scaled[i-60:i,0])
    y_train.append(training_set_scaled[i,0])
X_train, y_train = np.array(X_train), np.array(y_train)
