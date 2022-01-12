#date: 2022-01-12T17:06:52Z
#url: https://api.github.com/gists/e3531361880c0cbde3a4921fa1e130ca
#owner: https://api.github.com/users/haykaza

#split data into training and testing set
X_train_pca,X_test_pca,y_train_pca,y_test_pca = train_test_split(pd.DataFrame(output_pca), y, test_size = 0.20,random_state=0)

#fit logistic regression model on the training set
logistic_regression_pca = LogisticRegression().fit(X_train_pca,y_train_pca)

#predict labels on testing set and report classification outputs
y_pred_pca = logistic_regression_pca.predict(X_test_pca)
print(classification_report(y_test_pca, y_pred_pca))