#date: 2022-04-15T16:46:27Z
#url: https://api.github.com/gists/51bc877a8b47aa4ad5ba38558c2aaf95
#owner: https://api.github.com/users/ShubhashreeSur

#feature and target matrix
y=encoded_df.country_destination
X=encoded_df.drop('country_destination',axis=1)

#train-test split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,shuffle=False)
print("X_train shape:",X_train.shape)
print("X_test shape:",X_test.shape)

#standardization
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)