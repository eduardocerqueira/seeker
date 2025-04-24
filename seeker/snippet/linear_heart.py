#date: 2025-04-24T17:10:22Z
#url: https://api.github.com/gists/d83b3145d716802b72bce808c6159929
#owner: https://api.github.com/users/pranav-4019

# Step 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import model_selection, linear_model, metrics

# Step 2: Load the Dataset
df = pd.read_csv('heart.csv')
print("Data Overview:")
print(df.head())
print("\nDataset Info:")
print(df.info())
print("\nStatistical Description:")
print(df.describe())

# Convert 'yes'/'no' and other categorical columns using one-hot encoding
df_encoded = pd.get_dummies(df, drop_first=True)

df_encoded.head()

# Define features and target variable
#This line removes the column 'age' from the dataset to use the rest of the columns as features (X).
#These are the input variables used by the model to make predictions.
X = df_encoded.drop('age', axis=1) #features
# This is the output or label that we want to predict using the model.
# This line selects the 'age' column as the target variable (y) — the value we want to predict
y = df_encoded['age'] #target

# Step 4: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Train Linear Regression Model
model = LinearRegression()  #creates instance of linear regression class
model.fit(X_train, y_train)  #fit() is the method used to train the model on the provided training data

# Step 6: Make Predictions
y_pred = model.predict(X_test) 
#using the trained model to make predictions on the testing data.
# y_pred This is the variable where the predicted values

# Step 7: Evaluate the Model
mae = mean_absolute_error(y_test, y_pred)   # calculating differnec between actual value and predicted value (checking performance after traing model)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("R-squared (R²) Score:", r2)

sns.scatterplot(x=y_test, y=y_pred, label="Predicted vs Actual")  # Add label here
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label="Perfect Prediction Line")  # Add label here
plt.xlabel("Actual Age")
plt.ylabel("Predicted Age")
plt.title("Actual vs Predicted Age")
plt.legend()  # Now it knows what to show
plt.show()