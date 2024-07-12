#date: 2024-07-12T16:36:46Z
#url: https://api.github.com/gists/2f282e8fc34488ba150542033c9f2c82
#owner: https://api.github.com/users/iYadavVaibhav

from pyspark.sql import SparkSession

# Create a SparkSession
spark = SparkSession.builder \
    .appName("LocalModeExample") \
    .master("local") \
    .getOrCreate()

# Example DataFrame operation
data = [("Alice", 1), ("Bob", 2), ("Cathy", 3)]
columns = ["Name", "Value"]

# Create DataFrame
df = spark.createDataFrame(data, columns)

# Show DataFrame
df.show()

# Stop the SparkSession
spark.stop()
