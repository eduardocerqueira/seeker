#date: 2023-01-26T16:47:46Z
#url: https://api.github.com/gists/7b61e698fed986bba74aa7a45ce1ea84
#owner: https://api.github.com/users/taisazero

import time
import os

from pyspark.ml import Pipeline
from pyspark.ml.feature import RegexTokenizer, NGram, HashingTF, MinHashLSH
from pyspark.sql.functions import col
from spark_session_builder import build_spark_session

spark = build_spark_session("spark://cpu64-dy-c6i-16xlarge-1:7077", 32, 128)
db = spark.read.parquet("/fsx/shared/pilev2_parquet/StackExchange_ver4_non_local_dedupped/dataset.parquet").limit(1_000_000) # Stage 0 & 1
# db.show()

start = time.time()
# spark.sparkContext.defaultParallelism = os.cpu_count()
rdd = spark.sparkContext.parallelize(db.collect(), numSlices=5_000)
# Fit the pipeline to the parallelized data pipelineModel = pipeline.fit(rdd)

df = spark.createDataFrame(rdd, db.schema)
#, db.schema)

model = Pipeline(stages=[
    RegexTokenizer( # Stage 2
        pattern= "**********"="text", outputCol="tokens", minTokenLength=1
    ),
    NGram(n= "**********"="tokens", outputCol="ngrams"), # Stage 3
    HashingTF(inputCol="ngrams", outputCol="vectors"), # Stage 4
    MinHashLSH(inputCol="vectors", outputCol="lsh", numHashTables=13) # Stage 5
]).fit(df)

db_hashed = model.transform(df)
hashed_rdd = spark.sparkContext.parallelize(db_hashed.collect(), numSlices=5_000)
db_hashed = spark.createDataFrame(hashed_rdd, db_hashed.schema)

duplicates = model.stages[-1].approxSimilarityJoin(
    db_hashed,
    db_hashed,
    0.15,
    distCol="JaccardDistance"
).filter("datasetA.id < datasetB.id") # Stage 6
# duplicates.show()
duplicates.write.parquet("./duplicates", mode="overwrite") # Stage 7
end = time.time()
print(f"Time taken: {end - start} for {db.count()} rows")()} rows")