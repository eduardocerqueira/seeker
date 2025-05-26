#date: 2025-05-26T16:56:11Z
#url: https://api.github.com/gists/cd68e6989b11cf48e8d8b5ed0d306e5c
#owner: https://api.github.com/users/cuongtv312

import joblib
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

data = load_iris()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

joblib.dump(model, "iris_model.joblib")
print("Model saved as 'iris_model.joblib'")