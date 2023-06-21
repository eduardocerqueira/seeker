#date: 2023-06-21T16:35:49Z
#url: https://api.github.com/gists/cbd5cee1e6f3db5356e787467d2e5316
#owner: https://api.github.com/users/KrisAff84

import boto3        # Imports the boto3 module to allow interaction with AWS services
import json         # Imports the json module to allow ease of working with json data

def create_table(name, hash_key, range_key):      # creates the create_table function
    ddb = boto3.client('dynamodb')                # sets our interface to client
    response = ddb.create_table(                  # creates DynamoDB table
        TableName=name,                           # TableName set to variable name
        KeySchema=[                               # sets partition and sort keys, names are the variables
            {                                   
                'AttributeName': hash_key,        # hash_key and range_key that we pass as parameters to function
                'KeyType': 'HASH'
            },
            {
                'AttributeName': range_key,
                'KeyType': 'RANGE'
            }
        ],
        AttributeDefinitions=[                    # sets hashkey and rangekey type, in this case string(S
            {                                     
                'AttributeName': hash_key,
                'AttributeType': 'S',
            },
            {
                'AttributeName': range_key,
                'AttributeType': 'S',
            }
        ],
        BillingMode='PROVISIONED',                # billing method can be provisioned
        ProvisionedThroughput={                   # or on demand
            'ReadCapacityUnits': 5,
            'WriteCapacityUnits': 5
        },
        StreamSpecification={
            'StreamEnabled': False,
        },
        TableClass='STANDARD',  
    )
    print(json.dumps(response, indent=4, default=str)) # formats response in json format

def main():
    name = 'Songs'                               # parameter name set to 'Songs'
    hash_key = 'Singer'                          # parameter hash_key set to 'Singer'
    range_key = 'Song'                           # parameter range_keyh set to 'Song'
    create_table(name, hash_key, range_key)      # creates table with assigned parameters
  
  
if __name__ == '__main__':
    main()
    