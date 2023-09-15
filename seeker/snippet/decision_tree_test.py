#date: 2023-09-15T17:06:45Z
#url: https://api.github.com/gists/4a550107957698ab7af03dd10b4e664e
#owner: https://api.github.com/users/tuhdo

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import export_graphviz
import graphviz


# uncomment to choose dataset to train
dataset_path = 'add_dataset.csv'
# dataset_path = 'mul_dataset.csv'
df = pd.read_csv(dataset_path)

# Split the dataset into features and label
X = df[['Operand1', 'Operand2']]
y = df['Sum']

# Split the dataset into train set and val set
# On my machine, trained model did addition slightly wrong (e.g. 1 + 1 = 1.1) when test_size = 0.85
# Likewise, for multiplication, results stopped being accurate when test_size = 0.66.
test_size = 0.65 
random_state = 1
is_shuffle = True

X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=test_size,
    random_state=random_state,
    shuffle=is_shuffle
)

regressor = DecisionTreeRegressor(random_state=random_state)
regressor.fit(X_train, y_train)

# Train the model with random forest
# regressor = RandomForestRegressor(random_state=random_state)
# regressor.fit(X_train, y_train)

# Create a feature vector with your input data (operands)
# In this example, we're using [5, 7] for prediction
feature_vector = np.array([[353, 103]]) # add test
# feature_vector = np.array([[5, 15]]) # mul test

# Make a prediction
result = regressor.predict(feature_vector)

print(f"feature_vector: {feature_vector}")
print(f"The predicted sum is: {result[0]}")

# Uncomment below code to generate tree visualization with Graphviz.
# # Export the first decision tree to a DOT file
# tree = regressor.estimators_[0]  # Get the first decision tree from the forest
# dot_data = export_graphviz(tree, out_file=None,
#                            feature_names=["Operand1", "Operand2"],  # Replace with your feature names
#                            filled=True, rounded=True, special_characters=True)

# # Create a graph from the DOT data and display it
# graph = graphviz.Source(dot_data)
# graph.view(filename="add_dataset_sm")
