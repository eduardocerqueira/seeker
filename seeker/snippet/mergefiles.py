#date: 2023-12-28T16:41:40Z
#url: https://api.github.com/gists/2d18ca4bc5cec003ce75dc2b5443caa1
#owner: https://api.github.com/users/avgoncharova

import pandas as pd
import numpy as np
import os


# creating a data frame from column names
df = pd.DataFrame(columns = ['Employee Name', 'Employee ID', 'Employee State'])
# anything that is a csv file in the same directory as the code gets added to the data frame
for x in os.listdir():
    if x.endswith('.csv'):
        new_data= pd.read_csv(x)
        # the outer join allows for data union across different csv files as long as column names are the same
        df = pd.merge(df, new_data, how = 'outer')
# a new consolidated file will be created each time the code is run
df.to_csv('mergedcsv.csv',sep = ',',na_rep="NA", mode = 'w', index = False)