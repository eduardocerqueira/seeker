#date: 2024-09-17T16:45:08Z
#url: https://api.github.com/gists/58b059fb7f27264e9e563d064e058157
#owner: https://api.github.com/users/docsallover

# Load the fake news dataset
fake = pd.read_csv("data/Fake.csv")

# Load the true news dataset
true = pd.read_csv("data/True.csv")

# Print the shape of each dataset
print("Fake news dataset shape:", fake.shape)
print("True news dataset shape:", true.shape)