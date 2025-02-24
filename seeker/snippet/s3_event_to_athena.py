#date: 2025-02-24T16:54:16Z
#url: https://api.github.com/gists/09ced08f6b8ae7be8a585cecad90d728
#owner: https://api.github.com/users/LX5321

import boto3
import os
import time
import json
import uuid

# Initialize AWS clients
s3 = boto3.client('s3')
athena = boto3.client('athena')

# Constants
bucket_name = 'your-s3-bucket-name'
database_name = 'your_athena_database'
table_name = 'your_athena_table'
athena_output_bucket = f's3://{bucket_name}/athena-results/'

# Lambda Handler
def lambda_handler(event, context):
    operation = event.get('operation')  # e.g., 'upload', 'delete', 'rename', 'move'
    file_path = event.get('file_path')  # S3 file path (e.g., 'folder/myfile.csv')
    partition_column = 'year'  # Partition column in Athena
    partition_value = event.get('partition_value')  # e.g., 2025
    
    try:
        # 1. Upload a new file to S3 (CRUD operation: Upload)
        if operation == 'upload':
            file_content = event.get('file_content')  # Content of the file to upload (or generate mock data)
            s3.put_object(Bucket=bucket_name, Key=file_path, Body=file_content)
            print(f"Uploaded file to S3: s3://{bucket_name}/{file_path}")
            refresh_partition(partition_column, partition_value)
        
        # 2. Delete a file from S3 (CRUD operation: Delete)
        elif operation == 'delete':
            s3.delete_object(Bucket=bucket_name, Key=file_path)
            print(f"Deleted file from S3: s3://{bucket_name}/{file_path}")
            drop_and_recreate_partition(partition_column, partition_value)
        
        # 3. Rename a file in S3 (CRUD operation: Rename)
        elif operation == 'rename':
            new_file_path = event.get('new_file_path')  # New path for the renamed file
            s3.copy_object(Bucket=bucket_name, CopySource={'Bucket': bucket_name, 'Key': file_path}, Key=new_file_path)
            s3.delete_object(Bucket=bucket_name, Key=file_path)
            print(f"Renamed file from s3://{bucket_name}/{file_path} to s3://{bucket_name}/{new_file_path}")
            refresh_partition(partition_column, partition_value)
        
        # 4. Move a file in S3 (CRUD operation: Move)
        elif operation == 'move':
            new_file_path = event.get('new_file_path')  # New path for the file
            s3.copy_object(Bucket=bucket_name, CopySource={'Bucket': bucket_name, 'Key': file_path}, Key=new_file_path)
            s3.delete_object(Bucket=bucket_name, Key=file_path)
            print(f"Moved file from s3://{bucket_name}/{file_path} to s3://{bucket_name}/{new_file_path}")
            refresh_partition(partition_column, partition_value)
        
        else:
            raise ValueError("Invalid operation specified")
        
        return {
            'statusCode': 200,
            'body': json.dumps(f"Operation '{operation}' completed successfully.")
        }
    
    except Exception as e:
        print(f"Error: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps(f"Error: {str(e)}")
        }


# Function to refresh the Athena partition
def refresh_partition(partition_column, partition_value):
    print(f"Refreshing partition for {partition_column} = {partition_value} in Athena.")
    
    # Run the ALTER TABLE ADD PARTITION command for the partition column
    query = f"MSCK REPAIR TABLE {table_name};"
    execute_athena_query(query)
    
    print(f"Partition for {partition_column} = {partition_value} refreshed in Athena.")

# Function to drop and recreate the Athena partition
def drop_and_recreate_partition(partition_column, partition_value):
    print(f"Dropping and recreating partition for {partition_column} = {partition_value} in Athena.")
    
    # First, drop the partition
    drop_query = f"ALTER TABLE {table_name} DROP PARTITION ({partition_column}='{partition_value}');"
    execute_athena_query(drop_query)
    
    # Then, recreate the partition (assuming data exists for that partition)
    create_query = f"ALTER TABLE {table_name} ADD PARTITION ({partition_column}='{partition_value}') LOCATION 's3://{bucket_name}/{partition_column}={partition_value}/';"
    execute_athena_query(create_query)
    
    print(f"Partition for {partition_column} = {partition_value} dropped and recreated in Athena.")

# Function to execute Athena queries
def execute_athena_query(query):
    query_execution_id = athena.start_query_execution(
        QueryString=query,
        QueryExecutionContext={'Database': database_name},
        ResultConfiguration={
            'OutputLocation': athena_output_bucket
        }
    )['QueryExecutionId']
    
    # Wait for the query to finish
    state = 'RUNNING'
    while state in ['RUNNING', 'QUEUED']:
        time.sleep(5)  # Wait before checking query status
        result = athena.get_query_execution(QueryExecutionId=query_execution_id)
        state = result['QueryExecution']['Status']['State']
        print(f'Athena Query State: {state}')
    
    if state == 'SUCCEEDED':
        print(f"Query succeeded: {query}")
    else:
        print(f"Query failed: {query}")
