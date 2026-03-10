#date: 2026-03-10T17:38:51Z
#url: https://api.github.com/gists/3b3c188cb52b416e88405b1a18fdfce2
#owner: https://api.github.com/users/devreyakan

from micromlgen import port
from sklearn.svm import SVC
from sklearn.datasets import load_iris

if __name__ == '__main__':
    iris = load_iris()
    X = iris.data
    y = iris.target
    clf = SVC(kernel='linear', gamma=0.001).fit(X, y)
    print(port(clf))
