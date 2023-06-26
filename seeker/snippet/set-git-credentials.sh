#date: 2023-06-26T17:00:31Z
#url: https://api.github.com/gists/2a9c3db5e442359d4855a170df47c087
#owner: https://api.github.com/users/HarshadRanganathan

# Installs a git helper function which retrieves the password or developer token from Secrets Manager 
# directly for cloning a repository from a private git repo or pushing back changes upstream. 
# Storing passwords and tokens in Secrets Manager eliminates the need to store any sensitive information on EFS.

# Steps:
# 1. Add your password or personal developer token to Secret Manager
# 2. Set the secret name, key & email in the script below
# 3. Clone your repository via HTTP with the user name in the url, e.g. "git clone http://username@github.com/...."

#!/bin/bash

set -eux

## Parameters 
# your git provider, e.g. github.com
GIT_PROVIDER="github.com"
GIT_EMAIL_ADDRESS="<github_email_address>"

AWS_REGION="us-east-1"
# Secret name stored in AWS Secrets Manager
AWS_SECRET_NAME= "**********"
# Secret key name inside the secret
AWS_SECRET_KEY_GIT_USERNAME= "**********"
AWS_SECRET_KEY_GIT_PASSWORD= "**********"

## Script Body

PYTHON_EXEC=$(command -v python)

cat > ~/.aws-credential-helper.py <<EOL
#!$PYTHON_EXEC

import sys
import json
import boto3
import botocore

GIT_PROVIDER='$GIT_PROVIDER'
AWS_REGION='$AWS_REGION'
AWS_SECRET_NAME= "**********"
AWS_SECRET_KEY_GIT_USERNAME= "**********"
AWS_SECRET_KEY_GIT_PASSWORD= "**********"

if len(sys.argv) < 2 or sys.argv[1] != 'get':
    exit(0)

credentials = {}
for line in sys.stdin:
    if line.strip() == "":
        break
    key, value = line.split('=')[0:2]
    credentials[key.strip()] = value.strip()

if credentials.get('host', '') == GIT_PROVIDER:
    client = "**********"=AWS_REGION)
    try:
        response = "**********"=AWS_SECRET_NAME)
    except botocore.exceptions.ClientError as e:
        exit(1)
 "**********"  "**********"  "**********"  "**********"  "**********"i "**********"f "**********"  "**********"' "**********"S "**********"e "**********"c "**********"r "**********"e "**********"t "**********"S "**********"t "**********"r "**********"i "**********"n "**********"g "**********"' "**********"  "**********"i "**********"n "**********"  "**********"r "**********"e "**********"s "**********"p "**********"o "**********"n "**********"s "**********"e "**********": "**********"
        secret = "**********"
        secret_dict = "**********"
 "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"i "**********"f "**********"  "**********"A "**********"W "**********"S "**********"_ "**********"S "**********"E "**********"C "**********"R "**********"E "**********"T "**********"_ "**********"K "**********"E "**********"Y "**********"_ "**********"G "**********"I "**********"T "**********"_ "**********"U "**********"S "**********"E "**********"R "**********"N "**********"A "**********"M "**********"E "**********"  "**********"i "**********"n "**********"  "**********"s "**********"e "**********"c "**********"r "**********"e "**********"t "**********"_ "**********"d "**********"i "**********"c "**********"t "**********": "**********"
            credentials['username'] = "**********"
 "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"i "**********"f "**********"  "**********"A "**********"W "**********"S "**********"_ "**********"S "**********"E "**********"C "**********"R "**********"E "**********"T "**********"_ "**********"K "**********"E "**********"Y "**********"_ "**********"G "**********"I "**********"T "**********"_ "**********"P "**********"A "**********"S "**********"S "**********"W "**********"O "**********"R "**********"D "**********"  "**********"i "**********"n "**********"  "**********"s "**********"e "**********"c "**********"r "**********"e "**********"t "**********"_ "**********"d "**********"i "**********"c "**********"t "**********": "**********"
            credentials['password'] = "**********"

for key, value in credentials.items():
    print('{}={}'.format(key, value))

EOL

chmod +x ~/.aws-credential-helper.py
git config --global credential.helper ~/.aws-credential-helper.py
git config --global user.name "$AWS_SECRET_KEY_GIT_USERNAME"
git config --global user.email "$GIT_EMAIL_ADDRESS"
