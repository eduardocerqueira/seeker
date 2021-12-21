#date: 2021-12-21T16:58:28Z
#url: https://api.github.com/gists/927629e79cb6074a0623c18facaffcd7
#owner: https://api.github.com/users/pritishperpetua

import dask.dataframe as dd
import pandas as pd

test_df = pd.DataFrame({
    "col1": np.random.choice(a=["A", "B", "C"],  size=150000,  p=[0.5, 0.3, 0.2]) ,  
    "col2": np.random.normal(size = 150000), 
    "col3": np.random.normal(size = 150000), 
                       })


test_ddf = dd.from_pandas(test_df, npartitions = 15)

#####################################################################
# Using .groupby() directly on dask dataframe
#####################################################################
print(f"Number of partitions before groupby: {test_ddf.npartitions}")

agg_ddf = test_ddf.groupby("col1").agg(sum)

print(f"Number of partitions after groupby: {agg_ddf.npartitions}")

#####################################################################
# Using .groupby() with .map_partitions()
#####################################################################

print(f"Number of partitions before groupby: {test_ddf.npartitions}")

agg_ddf = test_ddf.map_partitions(lambda df: df.groupby("col1").agg(sum))

print(f"Number of partitions after groupby {agg_ddf.npartitions}")