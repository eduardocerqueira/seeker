#date: 2023-10-25T16:48:05Z
#url: https://api.github.com/gists/f624ccd89bc8bc443cf819b5b05f5de8
#owner: https://api.github.com/users/fruit-ninja

import logging
from awsglue import DynamicFrame

def write_partitioned_dynamic_frame(table, glue_context, dest_table_name, s3_target, dest_db_name, max_retries=5, base_wait_time=2):
    final_dynamicframe = DynamicFrame.fromDF(table, glue_context, "final_dynamicframe")
    partitions = table.select("partition_key").distinct().rdd.flatMap(lambda x: x).collect()

    for partition in partitions:
        retries = 0
        while retries < max_retries:
            try:
                # Filter dynamic frame for the specific partition
                partitioned_dynamicframe = final_dynamicframe.filter(f"partition_key = '{partition}'")
                
                # Check if the partition already exists in the destination (this logic can be adjusted based on your needs)
                if not check_partition_exists(s3_target, partition):  # Implement this function
                    covs_sink = glue_context.getSink(
                    )
                    covs_sink.setFormat("glueparquet")
                    covs_sink.setCatalogInfo(catalogDatabase=dest_db_name, catalogTableName=dest_table_name)
                    logging.info(f"Writing data for partition: {partition}")
                    covs_sink.writeFrame(partitioned_dynamicframe)
                break  # Exit the retry loop if write is successful
            except Exception as e:
                logging.error(f"Error occurred for partition {partition}: {e}. Retrying...")
                retries += 1
                time.sleep(base_wait_time * (2 ** retries))
        else:
            logging.error(f"Failed to write partition {partition} after {max_retries} retries.")

def check_partition_exists(s3_target, partition):
    # Implement logic to check if the given partition already exists in the destination.
    # This can be based on checking the S3 path or querying Glue Catalog.
    pass