#date: 2021-12-13T16:54:49Z
#url: https://api.github.com/gists/be1688847830c17514792818f815dedf
#owner: https://api.github.com/users/ifakat

# Preparing X_test and predicting the prices
X_test = []
for i in range(60,311):
    X_test.append(inputs[i-60:i,0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)
