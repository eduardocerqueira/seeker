#date: 2021-09-17T16:57:53Z
#url: https://api.github.com/gists/101e1acee38ef0920f72e4bbd07b5d1e
#owner: https://api.github.com/users/DC-gists

import pandas as pd

file_path = "C:/...path to your file/"

df_items = pd.read_csv(file_path + "items_sample.csv")
df_inventory_activity = pd.read_csv(file_path + "inventory_activity_sample.csv")

print("Items sample")
print(df_items.head())

print("Inventory activity sample")
print(df_inventory_activity.head())