#date: 2023-04-13T17:07:13Z
#url: https://api.github.com/gists/0a0f4b21847e032b1a45e032daa0aa81
#owner: https://api.github.com/users/jcohen66

import random
import numpy as np
import pandas as pd


# Create and populate a 5x2 NumPy array.
my_data = np.array([[0,3], [10,7], [20,9], [30,14], [40, 15]])

# Create a Python list that holds the names of the two columns.
my_column_names = ['temperature', 'activity']

# Create a DataFrame.
my_dataframe = pd.DataFrame(data=my_data, columns=my_column_names)

# Print the entire dataframe.
print(my_dataframe)

# Create a new column named adjusted.
my_dataframe["adjusted"] = my_dataframe["activity"] + 2

# Print the entire dataframe.
print(my_dataframe)

print("Rows #0, #1, and #2:")
print(my_dataframe.head(3), '\n')

# Use inner list.
print("Row #2:")
print(my_dataframe.iloc[[2]], '\n')

# Use list slicing.
print("Rows #1, #2, and #3:")
print(my_dataframe[1:4], '\n')

# Access like dictionary.
print("Column 'temperature':")
print(my_dataframe['temperature'])


def gen_num(seed):
    return random.randint(0, 100)

# Generate matrix of random ints 0-100 inclusive with a 3x4 shape.
data = np.random.randint(low=0, high=101, size=(3,4))

seed = 100

# Create a Python list that holds the names of the two columns.
my_column_names = ['Eleanor', 'Chidi', 'Tahani', 'Jason']

# Create and populate a 5x2 NumPy array.
my_data = np.array([[gen_num(seed),gen_num(seed),gen_num(seed),gen_num(seed)], 
                    [gen_num(seed),gen_num(seed),gen_num(seed),gen_num(seed)],
                    [gen_num(seed),gen_num(seed),gen_num(seed),gen_num(seed)],
                    [gen_num(seed),gen_num(seed),gen_num(seed),gen_num(seed)]])

# Create using the data matrix.
my_data = pd.DataFrame(data=data, columns=my_column_names)


# Create a DataFrame.
my_dataframe = pd.DataFrame(data=my_data, columns=my_column_names)

# Print the entire dataframe.
print(my_dataframe)

# Print the value in row #1 of Eleanor.
print(my_dataframe["Eleanor"][1])

# Create a new column named Janet.
my_dataframe["Janet"] = my_dataframe["Tahani"] + my_dataframe["Jason"]

print(my_dataframe)

# Create a referemce by assigning my_dataframe to a new variable.
print("Experiment with a reference:")
reference_to_df = my_dataframe

# Print the starting value of a particular cell.
print("  Starting value of df: %d" % my_dataframe['Jason'][1])
print("  Starting value of reference_to_df: %d\n" % reference_to_df['Jason'][1])

# Modify a cell in df.
my_dataframe.at[1, 'Jason'] = my_dataframe['Jason'][1] + 5
print("  Updated df: %d" % my_dataframe['Jason'][1])
print("  Updated reference_to_df: %d\n\n" % reference_to_df['Jason'][1])

# Create a true copy of my_dataframe.
print("Experiment with a true copy.")
copy_of_my_dataframe = my_dataframe.copy()

# Print the starting value of a particular cell.
print("  Starting value of df: %d" % my_dataframe['Jason'][1])
print("  Starting value of reference_to_df: %d\n" % copy_of_my_dataframe['Jason'][1])

my_dataframe.at[1, 'Jason'] = my_dataframe['Jason'][1] + 5

print("  Updated value of df: %d" % my_dataframe['Jason'][1])
print("  Updated value of reference_to_df: %d\n" % copy_of_my_dataframe['Jason'][1])



