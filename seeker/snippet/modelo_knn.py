#date: 2022-01-27T17:08:55Z
#url: https://api.github.com/gists/8d86e791fce91205a82878033e894108
#owner: https://api.github.com/users/sarahmbss

#modelo
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_treinamento, y_treinamento)

#obtenção das previsões
previsoes = knn.predict(X_teste)
previsoes