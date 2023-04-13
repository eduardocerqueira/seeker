#date: 2023-04-13T16:57:08Z
#url: https://api.github.com/gists/4f20a7a7a56bd3942ac1f26cc434d70a
#owner: https://api.github.com/users/tariqmassaoudi

import pandas as pd

# Load the Titanic dataset
titanic_df = pd.read_csv('titanic.csv')

# Use select_dtypes() to select only numerical columns
numerical_cols = titanic_df.select_dtypes(include='number')

# Print the selected numerical columns
numerical_cols.head(5)
