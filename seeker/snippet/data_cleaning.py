#date: 2022-11-14T17:07:30Z
#url: https://api.github.com/gists/aedd9734192b03c43968734133afb643
#owner: https://api.github.com/users/ashok49473

import pandas as pd
from gensim.utils import simple_preprocess

# Preprocess function
def preprocess(x):
    return " ".join(simple_preprocess(x))

# Reading 20k complaints from the dataset (to increase speed of execution)
df = pd.read_csv("ConsumerComplaints.csv",nrows=20_000)

# Apply preprocess to the Complaint text
df['Consumer complaint narrative'] = df['Consumer complaint narrative'].apply(preprocess)

# Store all the 20k complaints in as list of strings
docs = df['Consumer complaint narrative'].tolist()