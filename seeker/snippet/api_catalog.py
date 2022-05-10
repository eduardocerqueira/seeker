#date: 2022-05-10T16:58:27Z
#url: https://api.github.com/gists/ca6c8d2096cebd37458fbaf30eca18b7
#owner: https://api.github.com/users/leviandrade25

#Usage of API Catalog on scala shell
#writen in scala

#log on scala shell
spark-shell

#Showing avaiable databases
scala> spark.catalog.listDatabases.show(false)
+-------+---------------------+------------------------------------------------+
|name   |description          |locationUri                                     |
+-------+---------------------+------------------------------------------------+
|default|Default Hive database|hdfs://namenode:8020/user/hive/warehouse        |
|levi   |                     |hdfs://namenode:8020/user/hive/warehouse/levi.db|
+-------+---------------------+------------------------------------------------+

#seting the current database
spark.catalog.setCurrentDatabase("levi")

#listing tables
scala> spark.catalog.listTables.show()
+----------+--------+--------------------+---------+-----------+
|      name|database|         description|tableType|isTemporary|
+----------+--------+--------------------+---------+-----------+
|tab_alunos|    levi|                null|  MANAGED|      false|
|table_hive|    levi|                null|  MANAGED|      false|
|    titles|    levi|Imported by sqoop...|  MANAGED|      false|
+----------+--------+--------------------+---------+-----------+

#listing columns

scala> spark.catalog.listColumns("table_hive").show()
+----------+-----------+--------+--------+-----------+--------+
|      name|description|dataType|nullable|isPartition|isBucket|
+----------+-----------+--------+--------+-----------+--------+
|id_cliente|       null|     int|    true|      false|   false|
+----------+-----------+--------+--------+-----------+--------+

#see you