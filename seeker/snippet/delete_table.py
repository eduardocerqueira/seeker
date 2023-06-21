#date: 2023-06-21T16:35:49Z
#url: https://api.github.com/gists/cbd5cee1e6f3db5356e787467d2e5316
#owner: https://api.github.com/users/KrisAff84

import boto3
import json

def delete_table(table_name):        # Defines delete_table function
    dbb = boto3.client('dynamodb')   # DynamoDB interface assigned to client method
    response = dbb.delete_table(     # Deletes table
    TableName=table_name             # TableName assigned to parameter table_name
    )
    print(json.dumps(response, indent=2))    # Prints response in json format

def main():
    table_name = 'Songs'          # Table name to be deleted
    delete_table(table_name)
    
if __name__ == '__main__':
    main()
    