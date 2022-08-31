#date: 2022-08-31T17:18:07Z
#url: https://api.github.com/gists/c608a10550d72b53ef6785bc64520fa8
#owner: https://api.github.com/users/Daniel-Ozeas

import json
from pyspark.sql.types import *

# Define the schema
schema = StructType(
    [StructField("name", StringType(), True), StructField("age", IntegerType(), True)]
)

# Write the schema
with open("schema.json", "w") as f:
    json.dump(schema.jsonValue(), f)

# Read the schema
with open("schema.json") as f:
    new_schema = StructType.fromJson(json.load(f))
    print(new_schema.simpleString())
