#date: 2022-09-21T17:10:03Z
#url: https://api.github.com/gists/50a2a479d99cc10d62891f0dbf8aa974
#owner: https://api.github.com/users/Brillianttyagi

df_pyspark.select(['Name','Cars']).show()
# +---------+----+
# |     Name|Cars|
# +---------+----+
# |Deepanshu|  10|
# |      Ram|   8|
# |    Shyam|   4|
# |     Abhi|   3|
# |     Tony|   1|
# |    Stark|   2|
# +---------+----+

df_pyspark['Name']
# Column<'Name'>

df_pyspark.dtypes
#[('Name', 'string'), ('age', 'int'), ('Cars', 'int'), ('Salary', 'int')]