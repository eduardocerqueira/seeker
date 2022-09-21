#date: 2022-09-21T17:19:44Z
#url: https://api.github.com/gists/825cdc86086f9291d2690daf22f07b43
#owner: https://api.github.com/users/Brillianttyagi

### Adding Columns in data frame
df_pyspark=df_pyspark.withColumn('Age after three years',df_pyspark['age']+3)

df_pyspark.show()
# +---------+---+----+------+---------------------+
# |     Name|age|Cars|Salary|Age after three years|
# +---------+---+----+------+---------------------+
# |Deepanshu| 31|  10| 30000|                   34|
# |      Ram| 30|   8| 25000|                   33|
# |    Shyam| 29|   4| 20000|                   32|
# |     Abhi| 24|   3| 20000|                   27|
# |     Tony| 21|   1| 15000|                   24|
# |    Stark| 23|   2| 18000|                   26|
# +---------+---+----+------+---------------------+

