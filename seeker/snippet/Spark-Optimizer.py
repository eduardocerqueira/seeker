#date: 2025-10-15T17:12:04Z
#url: https://api.github.com/gists/3ba32cf39481b907a788e6934794b6a1
#owner: https://api.github.com/users/codecraker

from pyspark.sql import SparkSession
from pyspark.sql.functions import col

# Initialize Spark session with optimized config
spark = SparkSession.builder \
    .appName("FastETL") \
    .config("spark.sql.shuffle.partitions", "200") \
    .config("spark.executor.memory", "4g") \
    .getOrCreate()

# Load 100TB dataset (example: CSV)
df = spark.read.csv("s3://your-bucket/huge_dataset.csv")

# Optimize transformation: Filter and aggregate
result = df.filter(col("value") > 100) \
           .groupBy("category") \
           .agg({"amount": "sum"}) \
           .cache()  # Cache for speed

# Save output efficiently
result.write.parquet("s3://your-bucket/output/")

spark.stop()