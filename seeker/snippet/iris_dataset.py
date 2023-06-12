#date: 2023-06-12T17:01:07Z
#url: https://api.github.com/gists/ac6ee7bbe9cc61ff9940048ea6b322c0
#owner: https://api.github.com/users/ainomic-dev

import numpy as np
from sklearn.datasets import load_iris

# Load the IRIS dataset
data = load_iris()

# Print the feature names
print("Feature names:", data.feature_names)

# Print the target names
print("Target names:", data.target_names)

# Print the first 5 samples
print("First 5 samples:")
print(data.data[:5])
print(data.target[:5])
