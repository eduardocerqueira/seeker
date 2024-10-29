#date: 2024-10-29T17:03:42Z
#url: https://api.github.com/gists/ba1735bbf5dc7e0631c8796013efe067
#owner: https://api.github.com/users/docsallover

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Assuming you have your features (X) and target variable (y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)