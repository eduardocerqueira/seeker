#date: 2022-09-23T17:26:45Z
#url: https://api.github.com/gists/2fc3e6971a9479af1fefd89e733aff1d
#owner: https://api.github.com/users/garystafford

# copy jobs to spark container
docker cp apache_spark_examples/ ${SPARK_CONTAINER}:/home/

# establish an interactive session with the spark container
docker exec -it ${SPARK_CONTAINER} bash

# set environment variables used by jobs
export BOOTSTRAP_SERVERS="kafka:29092"
export TOPIC_PURCHASES="demo.purchases"

# run batch processing job
cd /home/apache_spark_examples/
spark-submit spark_batch_kafka.py

# run stream processing job
spark-submit spark_streaming_kafka.py