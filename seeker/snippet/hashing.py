#date: 2024-03-18T17:07:49Z
#url: https://api.github.com/gists/e1a026a0e546d61d2d426f1f1839b645
#owner: https://api.github.com/users/stacklikemind

import pandas as pd
import hashlib

# Define the hashing function using SHA-256
def hash_value(value):
    # Ensure the value is a string
    value = str(value)
    return hashlib.sha256(value.encode()).hexdigest()

# Load the CSV file into a DataFrame
input_filename = 'your_input_file.csv'
df = pd.read_csv(input_filename)

# Specify the column you want to hash
column_to_hash = 'column_name'  # Change to your column name

# Apply the hash function to the column
df[column_to_hash] = df[column_to_hash].apply(hash_value)

# Save the modified DataFrame to a new CSV file
output_filename = 'hashed_output.csv'
df.to_csv(output_filename, index=False)

print(f"Processed file saved as '{output_filename}'.")
