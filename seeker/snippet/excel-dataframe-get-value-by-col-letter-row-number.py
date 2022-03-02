#date: 2022-03-02T17:08:14Z
#url: https://api.github.com/gists/d809110f1713150d8bcb07d2fb2ba24b
#owner: https://api.github.com/users/rrhg

import string


col_names = list(string.ascii_uppercase)
col_range = 'A:Z'

sheet = pd.read_excel('myfile', sheet_name='sheet1', header=None, usecols=col_range, names=col_names)

sheet.index += 1 # start rows from 1 instead of 0

value1 = sheet['D'][7]
value2 = sheet['D'][8]