#date: 2022-03-28T16:53:31Z
#url: https://api.github.com/gists/f8ea08077af0ccc919916e9aa78a15ea
#owner: https://api.github.com/users/vipassanaecon

#import needed package
import pandas as pd

#read file and create dataframe
df = pd.read_csv(r"file_path.csv", encoding= 'unicode_escape')
df.head() #view head of dataframe

#convert timestamp/datetime/etc column... from object (or other) datatype to datetime datatype if needed. 
#longest process in this script if needed, otherwise ignore.
df['timestamp'] = df['timestamp'].apply(pd.Timestamp) 

#view datatypes of your columns if needed or if you want to validate the conversion worked as needed.
print(df.dtypes) 

#sort timestamp/datetime column
df.sort_values(by=['timestamp'], ascending=True) 

#create new sorted csv file
df.to_csv(r"file_path_sorted.csv")   
