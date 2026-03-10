#date: 2026-03-10T17:38:43Z
#url: https://api.github.com/gists/7cb8204dd5da15b2c7ab0ec93b5e7948
#owner: https://api.github.com/users/devreyakan

from sklearn.svm import SVC
from micromlgen import port

# örneklerinizi veri kümesi klasörüne yerleştirin
# dosya başına bir sınıf
# CSV formatında satır başına bir özellik vektörü

features, classmap = load_features('dataset/')
X, y = features[:, :-1], features[:, -1]
classifier = SVC(kernel='linear').fit(X, y)
c_code = port(classifier, classmap=classmap, platform='attiny')
print(c_code)
