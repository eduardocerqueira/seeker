#date: 2022-12-15T17:09:17Z
#url: https://api.github.com/gists/f851c527797fc46286b93d03fdc62ca5
#owner: https://api.github.com/users/yubozhao

from sklearn import svm
from sklearn import datasets

# Load training data
iris = datasets.load_iris()
X, y = iris.data, iris.target

# Model Training
clf = svm.SVC()
clf.fit(X, y)

# Save model to BentoML local model store
import bentoml
saved_model = bentoml.sklearn.save_model("iris_clf", clf)
print(f"Model saved: {saved_model}")