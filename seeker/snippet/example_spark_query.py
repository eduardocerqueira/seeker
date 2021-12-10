#date: 2021-12-10T16:55:36Z
#url: https://api.github.com/gists/39a63ac953c908e6f53a874ad7e301ca
#owner: https://api.github.com/users/duyttran

import pyspark.sql.functions as f
from pyspark.sql import DataFrame as SparkDF

def generate_table(motion_events: SparkDF) -> SparkDF:
    driver_km_df = motion_events.withColumn("driver_id", f.col("driver_id))\
        .withColumn("update_week", f.col("start_time") - timedelta(days=f.col("start_time").weekday()))\
        .withColumn("km_driven", f.col("veh_odo_end") - f.col("veh_odo_start"))\
        .groupby(["driver_id", "update_week"])["km_driven"]\
        .sum()
    return driver_km_df