#date: 2021-10-22T17:10:01Z
#url: https://api.github.com/gists/55a75aa95b40ac2a481c6a4b588966ea
#owner: https://api.github.com/users/patrickbrus

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

# import cleaned dataset and class names plus meta data data frame
df = pd.read_csv(r"data\Bird_Species_cleaned.csv")
df_classes = pd.read_csv(r"data\class_dict.csv")

# start with one-hot encoding the column "labels"
df_encoded = pd.get_dummies(df, columns=["labels"])
df_encoded.columns = ["filepaths", "data set", "class_index", *class_names]

# create stratified shuffle split object for creating one 80/20 split
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

# split in train and test
for train_index, test_index in sss.split(df_encoded.drop(columns=class_names), df_encoded[class_names]):
    print("TRAIN:", train_index, "TEST:", test_index)
    df_train = df_encoded.iloc[train_index]
    df_test = df_encoded.iloc[test_index]
    
# create a copy of the training dataframe     
df_train_tmp = df_train.copy()

# split the training dataframe further into training and validation data using a 80/20 split
for train_index, test_index in sss.split(df_train.drop(columns=class_names), df_train[class_names]):
    print("TRAIN:", train_index, "TEST:", test_index)
    df_train = df_train_tmp.iloc[train_index]
    df_valid = df_train_tmp.iloc[test_index]
    

# quickly check if number of samples of train + valid + test still matches the total number of samples
assert df_train.shape[0] + df_valid.shape[0] + df_test.shape[0] == df.shape[0], "Number of samples in splits does not match"    