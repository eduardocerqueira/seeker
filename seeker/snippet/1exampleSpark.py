#date: 2025-01-14T17:00:27Z
#url: https://api.github.com/gists/b8d5f544aa98d00a984dc0b8b1b56b5d
#owner: https://api.github.com/users/gonzalezgarciacristian

"""1exampleSpark"""
from pyspark.sql import SparkSession

# We use this file. Same route level as the script
file = "facebook-names-unique.txt"
# Creating the SparkSession
spark = SparkSession.builder.appName("FirstExample").getOrCreate()
# Loading the file
logData = spark.read.text(file).cache()

# Our logic
numAs = logData.filter(logData.value.contains('a')).count()
numBs = logData.filter(logData.value.contains('b')).count()

#Our results
print("We have : %i, with a and : %i with b" % (numAs, numBs))

# We stop spark
spark.stop()