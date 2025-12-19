#date: 2025-12-19T16:50:08Z
#url: https://api.github.com/gists/44546dd668064dd32535d898e5cb7471
#owner: https://api.github.com/users/sgouda0412

#### BASIC ########################################################################################################################

# cleaning str in the header 
df.columns = [x.lower().strip() for x in df.columns] # lower case, trim leading and trailing spaces
df.columns = [x.strip().replace(' ', '_') for x in df.columns] # replace whitespaces b/w words with _

# checking NaN in all df 
df.isnull().values.any()

# get column-slices
df.ix[:,:2] 
df.iloc[:,[0,3]] 
df.ix[:,'Column_name':] 

# header as a list 
list(df.columns.values)
df.columns.tolist() 

# unique values in a column
df['col_name'].unique()

# series to dataframe
df = pd.DataFrame( that_series, columns=['count'] )
# unique value and its counts in column 
df['col_name'].value_counts()

# plot top 10 value counts as a bar chart
df['col_name'].value_counts()[:10].plot(kind = 'bar', title="This is a title.")

# drop col by col name 
df = df.drop('column_name', 1)

# drop col by index 
df.drop(df.columns[[0, 1, 3]], axis=1) # Note: zero indexed

# drop columns based on a list (drop_col)
df.drop([col for col in drop_col if col in df], axis=1, inplace=True)

# reorder col
new_order = ['column1', 'column4', 'column2', 'column3']
df = df[new_order]

# rename col names 
df.columns = ['Name1', 'Name2', 'Name3'...]
df=df.rename(columns = {'two':'new_name'})


# output file without index
df.to_csv('example.csv', index=False)

# tsv file
df.to_csv('example.tsv', sep='\t')

# use a list of values to select rows, and put selected rows in a new df
ItemsYouWant = ['a', 'b', 'c']
df[df['old_column'].isin([ItemsYouWant])]

# dorp NaN based on one col
df = df[df['Things'].notnull()]

# drop duplicates
df = df.drop_duplicates('Things')

#split dictionary into columns 
df = pd.DataFrame({'a':[1,2,3], 'b':[{'c':1}, {'d':3}, {'c':5, 'd':6}]})
df['b'].apply(pd.Series)

################################## Time Date ################################################

# turn str to datetime formate
import pandas as pd
from datetime import datetime
from functools import partial
to_datetime_fmt = partial(pd.to_datetime, format= "%m/%d/%Y")
df['date'] = df['date'].apply(to_datetime_fmt)

################################## Formatting ################################################

pd.options.display.max_columns = 2000 # remove ellipsis
pd.options.display.max_rows = 2000 # remove ellipsis