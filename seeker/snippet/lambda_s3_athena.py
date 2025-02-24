#date: 2025-02-24T16:49:37Z
#url: https://api.github.com/gists/67bcb097a67b5c030fb1b717feb45e36
#owner: https://api.github.com/users/LX5321

import boto3
import csv
import io
import time
import json
import uuid

# AWS Clients
s3 = boto3.client('s3')
athena = boto3.client('athena')

# Lambda Handler
def lambda_handler(event, context):
    # Step 1: Create mock CSV data
    mock_data = [
        ['id', 'name', 'age'],
        ['1', 'Alice', '30'],
        ['2', 'Bob', '25'],
        ['3', 'Charlie', '35']
    ]

    # Convert data to CSV in memory
    csv_file = io.StringIO()
    writer = csv.writer(csv_file)
    writer.writerows(mock_data)
    csv_file.seek(0)  # Move to the beginning of the StringIO object

    # Step 2: Write CSV to S3
    bucket_name = 'your-s3-bucket-name'
    s3_key = f'mock-data/{uuid.uuid4()}.csv'
    
    # Upload CSV data to S3
    s3.put_object(Bucket=bucket_name, Key=s3_key, Body=csv_file.getvalue())
    print(f'Uploaded CSV file to s3://{bucket_name}/{s3_key}')

    # Step 3: Create Athena Query (Assuming a pre-existing Athena table)
    database_name = 'your_athena_database'
    table_name = 'your_athena_table'
    
    # Construct SQL Query
    query = f"SELECT * FROM {table_name} WHERE age > 25;"

    # Step 4: Start Athena Query
    query_execution_id = athena.start_query_execution(
        QueryString=query,
        QueryExecutionContext={'Database': database_name},
        ResultConfiguration={
            'OutputLocation': f's3://{bucket_name}/athena-results/{uuid.uuid4()}/'
        }
    )['QueryExecutionId']
    
    print(f'Started Athena query with Execution ID: {query_execution_id}')

    # Step 5: Wait for Query to Finish
    state = 'RUNNING'
    while state in ['RUNNING', 'QUEUED']:
        time.sleep(5)  # Wait before checking query status
        result = athena.get_query_execution(QueryExecutionId=query_execution_id)
        state = result['QueryExecution']['Status']['State']
        print(f'Query state: {state}')
    
    # Step 6: Check for Query Results
    if state == 'SUCCEEDED':
        result_location = result['QueryExecution']['ResultConfiguration']['OutputLocation']
        print(f'Query succeeded. Results are located at: {result_location}')
        
        # Step 7: Fetch the results
        result_object = s3.get_object(Bucket=bucket_name, Key=result_location.replace(f's3://{bucket_name}/', ''))
        result_data = result_object['Body'].read().decode('utf-8')
        print(f'Results: {result_data}')
        
        return {
            'statusCode': 200,
            'body': json.dumps('Query succeeded, check logs for details.')
        }
    else:
        print('Query failed.')
        return {
            'statusCode': 500,
            'body': json.dumps('Query failed.')
        }
