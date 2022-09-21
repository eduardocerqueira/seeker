#date: 2022-09-21T17:25:39Z
#url: https://api.github.com/gists/bd7c759a1c3ab7a6e3062162fa6d0dae
#owner: https://api.github.com/users/Brillianttyagi

### Rename the columns
#here we use shoe in the same line
df_pyspark.withColumnRenamed('Cars','Private Jet').show()
# +---------+---+-----------+------+
# |     Name|age|Private Jet|Salary|
# +---------+---+-----------+------+
# |Deepanshu| 31|         10| 30000|
# |      Ram| 30|          8| 25000|
# |    Shyam| 29|          4| 20000|
# |     Abhi| 24|          3| 20000|
# |     Tony| 21|          1| 15000|
# |    Stark| 23|          2| 18000|
# +---------+---+-----------+------+