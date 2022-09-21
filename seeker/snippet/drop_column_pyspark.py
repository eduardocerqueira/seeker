#date: 2022-09-21T17:22:20Z
#url: https://api.github.com/gists/86990e3987f8256859676aea44c7377c
#owner: https://api.github.com/users/Brillianttyagi

### Drop the columns
df_pyspark=df_pyspark.drop('Age after three years')

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