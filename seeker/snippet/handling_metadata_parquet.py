#date: 2025-08-08T17:06:11Z
#url: https://api.github.com/gists/c6e326e2712713ebd438bd37ad973b57
#owner: https://api.github.com/users/datavudeja

#%%
import pandas as pd
import numpy as np
import pyarrow as pa
import json

#%%
# Create a test dataframe
pd_df = pd.DataFrame({"col1": np.random.randint(10), "col2": np.random.rand(10)})


#%%
# Convert data frame into parquer or
# one can directly read csv into parquet too instead of converting
# pandas into parquet
pq_df = pa.Table.from_pandas(pd_df)

# parquet automatically infers the schema so:
print(pq_df.schema)

# also pandas sent some metadata to parquet
# usually metadata is empty dictionary if csv file is directly read inti parquet
print(pq_df.schema.metadata)

#%%
# Add custom metadata
# python dict and it has to be in byte string
custom_metadata = {
    b"iCAN": b'{"sumbitter": "Biranjan",\
  "email": "abc@gmail.com",\
    "psydo_req": "True",\
      "psydo_cols": ["col1"]}'
}

# too bad I can't use | to merge two dict running 3.8
merged_metadata = {**custom_metadata, **(pq_df.schema.metadata or {})}

pq_df = pq_df.replace_schema_metadata(merged_metadata)

# Check updated metadata
pq_df.schema.metadata

# Save the data as a file
pa.parquet.write_table(pq_df, "table_with_metadata.parquet")


# %%
# Processing metadata
# read the parquet file
# no need to read the whole data

pq_df1 = pa.parquet.read_metadata("table_with_metadata.parquet")
print(pq_df1.schema)
print(pq_df.schema.metadata)


# %%
## Since byte sting is bit unweildly so:

metadata = json.loads(pq_df.schema.metadata.get(b"iCAN"))
print(metadata)
metadata.get("sumbitter")
