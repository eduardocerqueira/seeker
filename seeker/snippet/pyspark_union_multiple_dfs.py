#date: 2022-01-12T17:16:28Z
#url: https://api.github.com/gists/845d1d3a749f617891ec4febc82997b3
#owner: https://api.github.com/users/cmpadden

from functools import reduce
from pyspark.sql import DataFrame, SparkSession

spark = SparkSession \
    .builder \
    .appName('Union DFs') \
    .getOrCreate()

df1 = spark.createDataFrame(
    [
        (1, "The"),
        (2, "Quick"),
    ],
    ['id', 'text']
)

df2 = spark.createDataFrame(
    [
        (3, "Brown"),
        (4, "Fox"),
    ],
    ['id', 'text']
)

df3 = spark.createDataFrame(
    [
        (5, "Jumped"),
        (6, "Over"),
    ],
    ['id', 'text']
)

reduce(DataFrame.union, [df1, df2, df3]).show()

# +---+------+
# | id|  text|
# +---+------+
# |  1|   The|
# |  2| Quick|
# |  3| Brown|
# |  4|   Fox|
# |  5|Jumped|
# |  6|  Over|
# +---+------+


