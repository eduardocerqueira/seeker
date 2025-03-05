#date: 2025-03-05T17:07:47Z
#url: https://api.github.com/gists/759e9a9fd0d72c8a3c9d33fc89ef510c
#owner: https://api.github.com/users/bianconif

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from mpl_ornaments.titles import set_title_and_subtitle

df = pd.read_csv('https://raw.githubusercontent.com/bianconif/graphic_communication_notebooks/refs/heads/master/data/top-10-meat-consumption-countries-2023.csv',comment='#')
print(df.head())