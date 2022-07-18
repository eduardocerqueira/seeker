#date: 2022-07-18T17:21:29Z
#url: https://api.github.com/gists/07ec1a897e879cb93d75e1890fce6b3f
#owner: https://api.github.com/users/zach-blumenfeld

./bin/spark-submit \
  --class org.neo4j.dwh.connector.Neo4jDWHConnector \
  --packages org.neo4j:neo4j-connector-apache-spark_2.12:4.1.0_for_spark_3,net.snowflake:spark-snowflake_2.12:2.10.0-spark_3.2,net.snowflake:snowflake-jdbc:3.13.15 \
  --master local
  /path/to/neo4j-dwh-connector-1.0.0-SNAPSHOT.jar \
  -p /path/to/dwh_job_config.json