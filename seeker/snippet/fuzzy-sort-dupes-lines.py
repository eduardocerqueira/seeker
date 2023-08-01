#date: 2023-08-01T16:55:16Z
#url: https://api.github.com/gists/54c501b9f93945d8380bc443d5b628d2
#owner: https://api.github.com/users/dzogrim

#!/opt/local/Library/Frameworks/Python.framework/Versions/3.9/bin/python3.9
"""Find similar duplicated lines in CSV file.

Needs py39-levenshtein, py39-fuzzywuzzy and py39-pandas.
"""
# -*- coding: utf-8 -*-

import pandas as pd
from fuzzywuzzy import fuzz

# load the csv file using semicolon as separator
df = pd.read_csv('shazam-to-clean.csv', sep=';', names=['Title', 'Artist'])

# process the text in both 'Title' and 'Artist',
# removing the characters you don't care about
df['clean_Title'] = df['Title'].str.replace("'", "")
df['clean_Artist'] = df['Artist'].str.replace("'", "")


# this function will be applied to every row
def is_duplicate(row):
    """Get duplicates."""
    # find rows with similar 'Artist'
    similar_rows = df[df['clean_Artist'] == row['clean_Artist']]

    # for each of these similar rows, check if the 'Title' is also similar
    for _, similar_row in similar_rows.iterrows():
        # here we check if the titles are similar
        # adjust the threshold as needed
        if fuzz.ratio(row['clean_Title'], similar_row['clean_Title']) > 105:
            return True
    return False

# apply the function to each row
df['duplicate_flag'] = df.apply(is_duplicate, axis=1)

# remove the duplicates
df = df[~df['duplicate_flag']]

# write the processed DataFrame into a new file
df.to_csv('shazam-dupes.csv', sep=';', index=False)
