#date: 2022-03-24T17:20:30Z
#url: https://api.github.com/gists/5884bac03a94638484cd717c8bf6a082
#owner: https://api.github.com/users/yz830620

import pandas as pd
import numpy as np


FILE_PATH = "data/movie_metadata.csv"
movies_table = pd.read_csv(FILE_PATH)
movies_table.head()