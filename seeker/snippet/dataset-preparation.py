#date: 2024-09-17T16:52:09Z
#url: https://api.github.com/gists/68d5b273d0473a793c5e192956037efd
#owner: https://api.github.com/users/docsallover

# Add a flag to track fake and real news
fake['target'] = 'fake'
true['target'] = 'true'

# Concatenate the dataframes
data = pd.concat([fake, true]).reset_index(drop=True)

# Check the shape of the combined dataset
print(data.shape)

# Shuffle the data
data = shuffle(data)
data = data.reset_index(drop=True)

# Check the first few rows of the shuffled data
print(data.head())