#date: 2021-12-13T16:54:45Z
#url: https://api.github.com/gists/416bfcf8fba86b10c99f6928ca5b8b15
#owner: https://api.github.com/users/ifakat

# Scaling the training set
sc = MinMaxScaler(feature_range=(0,1))
training_set_scaled = sc.fit_transform(training_set)
