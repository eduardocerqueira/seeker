#date: 2022-01-05T16:54:18Z
#url: https://api.github.com/gists/85562262e13cc7fbf679752457981fc3
#owner: https://api.github.com/users/ZhangMaKe

import json
import boto3
import os

COGNITO_USER_POOL_CLIENT_ID = os.environ['UserPoolClientId']

def handler(event, context):
    cognito_client = boto3.client('cognito-idp')
    username = event['Username']
    password = event['Password']

    try:
      sign_up_response = cognito_client.sign_up(
        ClientId=COGNITO_USER_POOL_CLIENT_ID,
        Username=username,
        Password=password
      )
    except Exception as e:
      print(f'Exception occurred when attemping to sign up user: {e}')

    print(f'Signup Response: {sign_up_response}')

    return {
        'statusCode': 200,
        'body': json.dumps('Hello from Lambda!')
    }