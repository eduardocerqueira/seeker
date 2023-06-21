#date: 2023-06-21T16:35:49Z
#url: https://api.github.com/gists/cbd5cee1e6f3db5356e787467d2e5316
#owner: https://api.github.com/users/KrisAff84

import boto3
import json

# Defines function and passes the various parameters to it
def delete_item(table, hash_key, hash_key_value, sort_key, sort_key_value):
    ddb = boto3.client('dynamodb')           # sets interface ddb to client
    response = ddb.delete_item(              # deletes item
        TableName=table,                  # TableName assigned to parameter table
        Key={
            hash_key: {                   # sets the name of the partition key
                'S': hash_key_value,      # sets the value of the partition key
            },
            sort_key: {                   # sets the name of the sort key
                'S': sort_key_value,      # sets the value of the sort key
            },
        },
    )
    print(json.dumps(response, indent=4, default=str))    # prints response in json format

def main():              # Assigns a value to the parameters and calls the function
    table = 'Songs'      
    hash_key = 'Singer'
    hash_key_value = 'Chris'
    sort_key = 'Song'
    sort_key_value = 'Brown Eyed Girl'
    delete_item(table, hash_key, hash_key_value, sort_key, sort_key_value) 
    
    
if __name__ == '__main__':
    main()
    