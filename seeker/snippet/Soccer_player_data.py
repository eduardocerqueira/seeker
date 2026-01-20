#date: 2026-01-20T17:07:44Z
#url: https://api.github.com/gists/aa94b81afca215e54b826fe7f8f8f428
#owner: https://api.github.com/users/ReeceL3

import pandas as pd

my_collection = {
    'player_num': [10, 7, 10, 10, 10, 11, 7],
    'first_name': ['Lionel', 'Cristiano', 'Edson', 'Neymar', 'Zlatan', 'Mohamed', 'George'],
    'last_name': ['Messi', 'Ronaldo', 'Pele', 'Jr', 'Ibrahimovic', 'Salah', 'Best'],
    'Goals': [672, 701, 767, 436, 511, 331, 179],
    'Reds': [3, 11, 0, 8, 9, 0, 8],
    'Yellows':[85, 129, 2, 97, 94, 18, 35],
    'date_joined': ['2004-10-16', '2002-10-07', '1956-09-07', '2009-03-07', '1999-03-31', '2010-05-03', '1963-09-14'],
}

df = pd.DataFrame(my_collection)
print(df)