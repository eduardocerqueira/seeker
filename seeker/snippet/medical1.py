#date: 2023-07-06T17:04:10Z
#url: https://api.github.com/gists/f2111fcb73f70cc1a2d901b89687a01f
#owner: https://api.github.com/users/5Spaak

# The link of our dataset
medical_charges_url = 'https://raw.githubusercontent.com/JovianML/opendatasets/master/data/medical-charges.csv'

# The urlretrieve function to put data in 'medical.csv' format
urlretrieve(medical_charges_url, 'medical.csv')

# Here, we're using pd.read_csv function to pull or read our data
medical_df = pd.read_csv('medical.csv')
