#date: 2022-01-27T17:07:54Z
#url: https://api.github.com/gists/552e81cd074856236feeb84f3f10e0d0
#owner: https://api.github.com/users/sarahmbss

#descrição dos dados, conseguimos analisar o max e min, média, variância dos valores numéricos
stats.describe(iris.data)

#criação dos previsores
previsores = iris.data
classe = iris.target

#divisão da base em treino e teste
X_treinamento, X_teste, y_treinamento, y_teste = train_test_split(previsores, classe, test_size = 0.3, random_state=0)