#date: 2022-07-05T17:14:16Z
#url: https://api.github.com/gists/87c7a009a76a7032cdb48b4c899f3b5c
#owner: https://api.github.com/users/kevinjnguyen

import base64

def lambda_handler(event, context):
  assert 'headers' in event, 'missing headers'
  assert 'authorization' in event['headers'], 'missing authorization header'
  authorization_header: str = event['headers']['authorization']
  authorization_tokens = authorization_header.split()
  assert len(authorization_tokens) == 2, 'malformed authorization bearer token'
  bearer_token = authorization_tokens[1]
  api_key = base64.b64decode(bearer_token).decode("utf-8")[:-1] # Remove the last character added from padding
  # Verify API Key