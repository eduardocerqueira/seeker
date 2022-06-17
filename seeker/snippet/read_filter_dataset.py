#date: 2022-06-17T16:52:22Z
#url: https://api.github.com/gists/2252e20b370235db4ee851da5cd78d4c
#owner: https://api.github.com/users/lucasazevedopassos

import pandas as pd
import yaml

# Read full dataset
df = pd.read_csv('Dataset/article1.csv')

# Open the YAML and filter by the query attribute
with open('App/YAML/cnn.yaml', 'r') as yamlFile:
    doc = yaml.load(yamlFile, Loader=yaml.FullLoader)
    cnn = df.query(doc['query'])