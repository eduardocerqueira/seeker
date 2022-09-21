#date: 2022-09-21T17:00:28Z
#url: https://api.github.com/gists/52eb68027031b038e1b07d07cefbc670
#owner: https://api.github.com/users/Brillianttyagi

## read the dataset
df_pyspark=spark.read.option('header','true').csv('details.csv',inferSchema=True)
### Check the schema
df_pyspark.printSchema()
# root
#  |-- Name: string (nullable = true)
#  |-- age: integer (nullable = true)
#  |-- Cars: integer (nullable = true)
#  |-- Salary: integer (nullable = true)

df_pyspark=spark.read.csv('details.csv',header=True,inferSchema=True)
df_pyspark.show()
# +---------+---+----+------+
# |     Name|age|Cars|Salary|
# +---------+---+----+------+
# |Deepanshu| 31|  10| 30000|
# |      Ram| 30|   8| 25000|
# |    Shyam| 29|   4| 20000|
# |     Abhi| 24|   3| 20000|
# |     Tony| 21|   1| 15000|
# |    Stark| 23|   2| 18000|
# +---------+---+----+------+