#date: 2025-08-08T17:10:22Z
#url: https://api.github.com/gists/56d5cbbd9dc2a7e6f821053411b03c91
#owner: https://api.github.com/users/datavudeja

'''
- Purpose of this code is to provide a custom group_by() function that will 
  keep NaN values in the output as a group.
- Currently, the Pandas function pd.DataFrame.groupby() will drop NaN's from the set of groups, 
  and I don't see a native way around this.
- group_by() is meant to be used as a drop-in replacement for pd.DataFrame.groupby(), 
  see examples below under the header ## Vignette/examples
- group_by() can take all of the same optional input arguments as the Pandas version. 
  It can also group by multiple columns.
- group_by() is fully backward compatible with our data set that contain "NA" strings.

'''

from pandas.core.frame import DataFrame
from numpy import nan

## Custom function definition
def group_by(df, cols, **kwargs):
    # Input validations.
    if not isinstance(df, DataFrame):
        raise ValueError("arg 'df' must be a pandas data frame")
    if not isinstance(cols, str) and not isinstance(cols, list):
        raise ValueError("arg 'cols' must be column headers to group by, as a single string or a list of strings")
    if isinstance(cols, str):
        cols = [cols]
    if not all([n in df.columns for n in cols]):
        missings = ", ".join([n for n in cols if n not in df.columns])
        raise ValueError("All input 'cols' must appear as columns in 'df'\n"\
                         "The following input col strings are not columns in df:\n"\
                         "%s" % missings)
    
    # Perform grouping operation.
    res = df.groupby([df[col].replace(nan, "NaN") for col in cols], **kwargs)
    
    return res


## Vignette/examples

import pandas as pd
import numpy as np

# Create test data frame.
df = pd.DataFrame({
    "fixed": [1, 1, 1, 1, 1, 1], 
    "color": ["brown", "orange", "brown", "orange", "brown", "orange"], 
    "animal": [np.nan, "dog", "cat", "cat", "dog", "cat"]
})

# Doing a groupby() operation on data that contains NaN values, the missing 
# values are dropped during the call to groupby().
test = df.groupby("animal", as_index = False)["fixed"].agg([np.sum])
print(test)
'''
        sum
animal     
cat       3
dog       2
'''

# If we use the custom group_by() function on the same data set that contains 
# NaN values, it will preserve the missing values and list them in the 
# summary stats table.
test = group_by(df, "animal", as_index = False)["fixed"].agg([np.sum])
print(test)
'''
        sum
animal     
NaN       1
cat       3
dog       2
'''

# group_by() is designed to take all of the same optional args as pandas.DataFrame.groupby().
# This example is using args "as_index" and "squeeze".
test = group_by(df, "animal", as_index = False, squeeze = True)

# group_by() can also take multiple column headers as args.
# Add "color" as an example.
test = group_by(df, ["animal", "color"], as_index = False)["fixed"].agg([np.sum])
print(test)
'''
               sum
animal color      
NaN    brown     1
cat    brown     1
       orange    2
dog    brown     1
       orange    1
'''