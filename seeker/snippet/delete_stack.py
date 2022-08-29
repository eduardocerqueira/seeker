#date: 2022-08-29T16:52:48Z
#url: https://api.github.com/gists/3f20e11f139618bb4b3200bdb6f649bb
#owner: https://api.github.com/users/ftestini

import json
import boto3

def lambda_handler(event, context):
    cf_client = boto3.client('cloudformation')
    cf_client.delete_stack(
        StackName='VPC-Endpoints'
    )
    return {
        'statusCode': 200,
        'body': json.dumps('Stack deletion launched!!')
    }
