#date: 2025-01-03T17:12:15Z
#url: https://api.github.com/gists/4504efb00c378a8c782316146937d92b
#owner: https://api.github.com/users/PieroPaialungaAI

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

X =  X_PCA[:,1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Linear Regression model
model = LinearRegression()

# Fit the model on the training data
model.fit(X_train.reshape(-1,1), y_train)

# Make predictions on the testing data
y_pred = model.predict(X_test.reshape(-1,1))

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R-squared:", r2)

# Coefficients of the model
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)

# Example of using the model to predict a new data point