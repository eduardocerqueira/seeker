#date: 2023-03-07T17:05:00Z
#url: https://api.github.com/gists/b8fd02aec267ec0765b9411a3491747d
#owner: https://api.github.com/users/DeepJani05

# importing  modules for project

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV

#Target
x = New_bonder.drop("LoanStatus",axis = 1)
y = New_bonder['LoanStatus']

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33 , stratify = y)
print(x_train.shape,y_train.shape)
print(x_test.shape,y_test.shape)


'''he simplest way to evaluate this model is using accuracy; we check the predictions against the actual values in the test set and count up how many the model got right.'''

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

#Hyperparameter Tuning
from scipy.stats import randint
param_dist = {'n_estimators': randint(50,500),
              'max_depth': randint(1,20)}

# Create a random forest classifier
rf = RandomForestClassifier()

# Use random search to find the best hyperparameters
rand_search = RandomizedSearchCV(rf,
                                 param_distributions = param_dist,
                                 n_iter=5,
                                 cv=5)

# Fit the random search object to the data
fnal=rand_search.fit(x_train, y_train)
print(fnal)

# Create a variable for the best model
best_rf = rand_search.best_estimator_

# Print the best hyperparameters
print('Best hyperparameters:',  rand_search.best_params_)

#More Evaluation Metrics

y_pred = best_rf.predict(x_test)

# Create the confusion matrix
cm = confusion_matrix(y_test, y_pred)

ConfusionMatrixDisplay(confusion_matrix=cm).plot()
plt.show()

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)

# Create a series containing feature importances from the model and feature names from the training data
feature_importances = pd.Series(best_rf.feature_importances_, index=x_train.columns).sort_values(ascending=False)

# Plot a simple bar chart
feature_importances.plot.bar()
plt.show()