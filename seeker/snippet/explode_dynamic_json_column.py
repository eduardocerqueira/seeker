#date: 2021-09-24T16:59:53Z
#url: https://api.github.com/gists/37964e2b83ebe2b746e5d20ff16a1018
#owner: https://api.github.com/users/FelipeRando

#When you have a dynamic column inside your Pyspark Dataframe, you can use below code to explode it's columns

columnX = spark.read.json(df.rdd.map(lambda row: row.dynamic_json_column)).drop('_corrupt_record')
for c in set(columnX.columns):
    df = df.withColumn(f'columnX_{c}',df[f'columnX.{c}'])
    
#explanation
#In line 3 we read our column as JSON Dataframe (with inferred schema)
#then we add a column in the main Dataframe for each key inside our JSON Column
#This works if we don't know the JSON fields in advance and yet, need to explode it's columns