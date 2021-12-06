#date: 2021-12-06T16:58:44Z
#url: https://api.github.com/gists/2d0944aee3fb71e22a071ffe721bc6f3
#owner: https://api.github.com/users/keitazoumana

from tabula import read_pdf
from tabulate import tabulate
import pandas as pd
import io

# Read the only the page n°6 of the file
food_calories = read_pdf('./data/food_calories.pdf',pages = 6, 
                         multiple_tables = True, stream = True)

# Transform the result into a string table format
table = tabulate(food_calories)

# Transform the table into dataframe
df = pd.read_fwf(io.StringIO(table))

# Save the final result as excel file
df.to_excel("./data/food_calories.xlsx")