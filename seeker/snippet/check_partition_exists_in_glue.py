#date: 2023-10-25T16:42:15Z
#url: https://api.github.com/gists/dae124099ef402edbfadd586652c7ca8
#owner: https://api.github.com/users/fruit-ninja

import boto3

def check_partition_exists_in_glue(dest_db_name, dest_table_name, partition):
    glue = boto3.client('glue')
    try:
        response = glue.get_partition(
            DatabaseName=dest_db_name,
            TableName=dest_table_name,
            PartitionValues=[partition]  # Adjust this if you have multiple partition columns
        )
        return True if 'Partition' in response else False
    except glue.exceptions.EntityNotFoundException:
        return False