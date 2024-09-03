#date: 2024-09-03T16:56:27Z
#url: https://api.github.com/gists/4e27bb8bae3bbff729866e1bc5866f8c
#owner: https://api.github.com/users/SeanLikesData

config = SedonaContext.builder() .\
    config('spark.jars.packages',
           'org.apache.sedona:sedona-spark-3.4_2.12:1.6.0,'
           'org.datasyslab:geotools-wrapper:1.6.0-28.2,'
           'uk.co.gresearch.spark:spark-extension_2.12:2.11.0-3.4'). \
    getOrCreate()

sedona = SedonaContext.create(config)