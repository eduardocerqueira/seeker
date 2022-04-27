#date: 2022-04-27T17:00:47Z
#url: https://api.github.com/gists/7bcefdb9e1e8038c6ca1d8ba0a0b27f1
#owner: https://api.github.com/users/jaochin

# KNN with general best features
knn = KNeighborsClassifier()

pipe_knn = Pipeline([('prepro',preproc_X_interest), ('knn',knn)])
param_grid = {'knn__n_neighbors':[5,9,15,19,25,29]}

knn_dft_best = GridSearchCV(pipe_knn, param_grid=param_grid, cv=5, scoring='recall')
knn_dft_best.fit(X_train[col_interest], y_train_tran)

predict_knn_dft_best = knn_dft_best.predict(X_test[col_interest])

print("Best Param :", knn_dft_best.best_params_)
print("Train Accuracy :", knn_dft_best.best_score_)
print("Test Accuracy :", accuracy_score(y_test_tran, predict_knn_dft_best), '\n')
print("Classification Report :\n", classification_report(y_test_tran, predict_knn_dft_best), '\n')