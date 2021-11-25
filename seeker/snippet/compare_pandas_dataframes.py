#date: 2021-11-25T17:12:33Z
#url: https://api.github.com/gists/a95ac760db3554f34320c357b02638d0
#owner: https://api.github.com/users/curiousest

# Because I couldn't get this to do what I wanted: 
# https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.compare.html

import pandas as pd
from typing import Optional, List

def compare_dataframes(df1, df2, drop_indexes=False, ignore_columns: Optional[List]=None):
    '''Returns two dataframes: the rows that only show up in each dataframe'''
    if drop_indexes:
        df1 = df1.reset_index(drop=True)
        df2 = df2.reset_index(drop=True)
    if ignore_columns:
        df1 = df1.drop(columns=ignore_columns, errors="ignore")
        df2 = df2.drop(columns=ignore_columns, errors="ignore")

    merged = df1.merge(df2, how="outer", indicator=True, suffixes=["_old", "_new"])

    df1_only = merged[merged["_merge"] == "left_only"]
    df1_only = df1_only.drop(columns=["_merge"])
    df2_only = merged[merged["_merge"] == "right_only"]
    df2_only = df2_only.drop(columns=["_merge"])
    return df1_only, df2_only