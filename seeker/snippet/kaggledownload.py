#date: 2023-08-30T16:57:59Z
#url: https://api.github.com/gists/800ad524e01b995279f617246e0b12d4
#owner: https://api.github.com/users/nilotpalc


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


#import requests
import os

# Replace 'username' and 'api_key' with your own Kaggle username and API key
username = 'USERNAME'
api_key = 'my_api_key'

# Set the Kaggle API credentials
os.environ['KAGGLE_USERNAME'] = username
os.environ['KAGGLE_KEY'] = api_key

# Download the dataset using the Kaggle API
!kaggle competitions download -c playground-series-s3e20

"""
# extract all the csv files from the zip archive and save them as individual dataframe
# save all the dataframes in a dictionary with the name of the csv file as the key
"""
from zipfile import ZipFile
zf = ZipFile('playground-series-s3e20.zip')
dfs = {}
for text_file in zf.infolist():
    if text_file.filename.endswith('.csv'):
        dfs[text_file.filename] = pd.read_csv(zf.open(text_file.filename, 'r'),encoding_errors='ignore')