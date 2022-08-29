#date: 2022-08-29T16:51:55Z
#url: https://api.github.com/gists/f09a49a84e12555394e0daeb5d6cc68a
#owner: https://api.github.com/users/ftestini

import json
import boto3

def lambda_handler(event, context):
    cf_client = boto3.client('cloudformation')
    cf_client.create_stack(
        StackName='VPC-Endpoints',
        TemplateURL='https://{FULL_TEMPLATE_URL}'
    )
    return {
        'statusCode': 200,
        'body': json.dumps('Stack creation launched!!')
    }