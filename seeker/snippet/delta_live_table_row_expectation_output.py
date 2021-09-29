#date: 2021-09-29T16:53:02Z
#url: https://api.github.com/gists/5cf464b17bf4bee66d912ffdbc6c1353
#owner: https://api.github.com/users/gbraccialli-db

pipelines_id = "88c2c93d-ac97-48f8-bd5e-2addd66ec781"

from pyspark.sql import functions as F

# Define DQ Expectations schema
schema = F.schema_of_json("""
  {"flow_progress":{
    "status":"status",
    "metrics":{"num_output_rows":-1},
    "data_quality":{"dropped_records":-1,
    "expectations":[
      {"name":"name",
       "dataset":"dataset",
       "passed_records":-1,
       "failed_records":-1}
     ]}}
  }""")      


df_dlt_output = (
  spark
  .read
  .load(f"dbfs:/pipelines/{pipelines_id}/system/events/")
  .withColumn("details_json", F.from_json("details", schema))
  .withColumn("data_quality", F.explode(
                                F.coalesce(
                                  "details_json.flow_progress.data_quality.expectations",
                                  F.array(F.struct(
                                    F.col("origin.flow_name").alias("dataset"),
                                    F.lit(-1).alias("failed_records"),
                                    F.lit("none").alias("name"),
                                    F.lit(-1).alias("passed_records")
                                  ))
                                )
                               )
  )
  .filter("event_type = 'flow_progress'")
  .filter("details_json.flow_progress.status = 'COMPLETED'")
  .select(
    "timestamp",
    "origin.flow_name",
    F.col("details_json.flow_progress.metrics.num_output_rows").alias("flow_num_output_rows"),
    F.col("details_json.flow_progress.data_quality.dropped_records").alias("flow_dropped_rows"),
    F.col("data_quality.name").alias("expectation_name"),
    F.col("data_quality.passed_records").alias("expectation_passed_records"),
    F.col("data_quality.failed_records").alias("expectation_failedrecords")
  )
  .sort("timestamp")
)

display(df_dlt_output)