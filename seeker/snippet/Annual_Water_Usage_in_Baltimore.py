#date: 2021-12-23T16:58:35Z
#url: https://api.github.com/gists/328eaf179dcd4019fda8956a103b308a
#owner: https://api.github.com/users/tqrahman

# Reading in the data
df = pd.read_csv('./data/yearly-water-usage.csv', index_col=0, parse_dates=True)

# Getting the splitting point
split_point = df.shape[0] - 10

# Splitting the data into train and test set
train, test = df[:split_point], df[split_point:]

# Viewing the dimensions of the train and test set
print(f'Train: {train.shape}, Test: {test.shape}')