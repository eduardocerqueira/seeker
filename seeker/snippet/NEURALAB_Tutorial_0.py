#date: 2022-12-19T16:53:07Z
#url: https://api.github.com/gists/5fac6e40a52ee5120b06527be53a043f
#owner: https://api.github.com/users/Biuni

# Google Colab: https://colab.research.google.com/drive/162MWdKLv-l2C-Wx57Opgz-WvzWK8eZNa?usp=sharing
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

digits = load_digits()
X = digits.images.reshape((len(digits.images), -1))
y = digits.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
mlp = MLPClassifier(hidden_layer_sizes=(15,), activation='logistic', alpha=1e-4, solver='sgd', tol=1e-4, learning_rate_init=.1, random_state=1, verbose=False)
predictions = mlp.predict(X_test)

print('Il modello ha un\'accuratezza del', round(accuracy_score(y_test, predictions) * 100, 2), '%')