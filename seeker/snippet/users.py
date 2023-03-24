#date: 2023-03-24T16:55:31Z
#url: https://api.github.com/gists/6f49286534d2757c85f13981681ae896
#owner: https://api.github.com/users/Arthurobo

# Age Range 18 - 25
import pandas as pd
import numpy as np

# Define the file path and delimiter
file_path = 'users.dat'
delimiter = '::'

# Define the data types for each column
dtype = np.dtype([
    ('user_id', 'i4'), 
    ('gender', 'U1'), 
    ('age', 'i4'), 
    ('occupation', 'i4'), 
    ('zip_code', 'U5')
])

# Load the data into a structured NumPy array
data = np.genfromtxt(file_path, delimiter=delimiter, dtype=dtype)

# Convert the NumPy array to a Pandas DataFrame
df = pd.DataFrame(data)

# Filter the data for individuals between the ages of 18 and 25
age_filter = (df['age'] >= 18) & (df['age'] <= 25)
filtered_data = df[age_filter]

# Print the filtered data
print(filtered_data)










# Age Range 25 - 34
import pandas as pd
import numpy as np

# Define the file path and delimiter
file_path = 'users.dat'
delimiter = '::'

# Define the data types for each column
dtype = np.dtype([
    ('user_id', 'i4'), 
    ('gender', 'U1'), 
    ('age', 'i4'), 
    ('occupation', 'i4'), 
    ('zip_code', 'U5')
])

# Load the data into a structured NumPy array
data = np.genfromtxt(file_path, delimiter=delimiter, dtype=dtype)

# Convert the NumPy array to a Pandas DataFrame
df = pd.DataFrame(data)

# Filter the data for individuals between the ages of 18 and 25
age_filter = (df['age'] >= 25) & (df['age'] <= 34)
filtered_data = df[age_filter]

# Print the filtered data
print(filtered_data)








# Age Range 35 - 44
import pandas as pd
import numpy as np

# Define the file path and delimiter
file_path = 'users.dat'
delimiter = '::'

# Define the data types for each column
dtype = np.dtype([
    ('user_id', 'i4'), 
    ('gender', 'U1'), 
    ('age', 'i4'), 
    ('occupation', 'i4'), 
    ('zip_code', 'U5')
])

# Load the data into a structured NumPy array
data = np.genfromtxt(file_path, delimiter=delimiter, dtype=dtype)

# Convert the NumPy array to a Pandas DataFrame
df = pd.DataFrame(data)

# Filter the data for individuals between the ages of 18 and 25
age_filter = (df['age'] >= 35) & (df['age'] <= 44)
filtered_data = df[age_filter]

# Print the filtered data
print(filtered_data)
