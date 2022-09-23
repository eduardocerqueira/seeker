#date: 2022-09-23T17:19:54Z
#url: https://api.github.com/gists/1581254bb705c52801505d95f2921ae3
#owner: https://api.github.com/users/garystafford

# establish an interactive session with the spark container
SPARK_CONTAINER=$(docker container ls --filter  name=streaming-stack_spark.1 --format "{{.ID}}")
docker exec -it -u 0 ${SPARK_CONTAINER} bash

# update and install wget
apt-get update && apt-get install wget -y

# install required job dependencies
wget https://repo1.maven.org/maven2/org/apache/commons/commons-pool2/2.11.1/commons-pool2-2.11.1.jar
wget https://repo1.maven.org/maven2/org/apache/kafka/kafka-clients/2.8.1/kafka-clients-2.8.1.jar
wget https://repo1.maven.org/maven2/org/apache/spark/spark-sql-kafka-0-10_2.12/3.3.0/spark-sql-kafka-0-10_2.12-3.3.0.jar
wget https: "**********"
mv *.jar /opt/bitnami/spark/jars/

exit
provider-kafka-0-10_2.12/3.3.0/spark-token-provider-kafka-0-10_2.12-3.3.0.jar
mv *.jar /opt/bitnami/spark/jars/

exit
