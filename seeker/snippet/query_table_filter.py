#date: 2023-06-21T16:35:49Z
#url: https://api.github.com/gists/cbd5cee1e6f3db5356e787467d2e5316
#owner: https://api.github.com/users/KrisAff84

import boto3
import json

def query_table_filter(singer, review='Review'):    # defines function with paramaters singer and review(with default set to 'Review')
    ddb = boto3.client('dynamodb')                  # sets ddb interface to client            
    response = ddb.query(                           # queries table
        TableName='Songs',                          # sets TableName to 'Songs'
        KeyConditionExpression='Singer = :singer',  # defines the key the query is based on
        FilterExpression= 'Review = :review',       # defines the attribute the filter is based on
        ExpressionAttributeValues={               # defines the values we query and filter by
            ':singer': {                          # sets the value of the partition key we query by
                'S': singer,
            },
            ':review': {                          # sets the value of the filter expression attribute
                'S': review,
            },
        },
    )
    
    print(json.dumps(response, ensure_ascii=False, indent=4))       # prints response formatted in json
    
    
def main():
    singer = 'Nathalie'                         # sets the parameter 'singer' to 'Nathalie'
    query_table_filter(singer)                  # query table with singer parameter passed in

if __name__ == '__main__':
    main()
    