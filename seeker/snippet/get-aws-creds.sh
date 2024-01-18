#date: 2024-01-18T17:05:58Z
#url: https://api.github.com/gists/f57a7a66dd8d841db1fae29d862d4513
#owner: https://api.github.com/users/radicaldo

#!/bin/bash

# This uses MFA devices to get temporary (eg 12 hour) credentials.  Requires
# a TTY for user input.
#
# GPL 2 or higher

if [ ! -t 0 ]
then
  echo Must be on a tty >&2
  exit 255
fi

identity=$(aws sts get-caller-identity)
username=$(echo -- "$identity" | sed -n 's!.*"arn:aws:iam::.*:user/\(.*\)".*!\1!p')
if [ -z "$username" ]
then
  echo "Can not identify who you are.  Looking for a line like
    arn:aws:iam::.....:user/FOO_BAR
but did not find one in the output of
  aws sts get-caller-identity

$identity" >&2
  exit 255
fi

echo You are: $username >&2

mfa=$(aws iam list-mfa-devices --user-name "$username")
device=$(echo -- "$mfa" | sed -n 's!.*"SerialNumber": "\(.*\)".*!\1!p')
if [ -z "$device" ]
then
  echo "Can not find any MFA device for you.  Looking for a SerialNumber
but did not find one in the output of
  aws iam list-mfa-devices --username \"$username\"

$mfa" >&2
  exit 255
fi

echo Your MFA device is: $device >&2

echo -n "Enter your MFA code now: " >&2
read code

tokens= "**********"

secret=$(echo -- "$tokens" | sed -n 's!.*"SecretAccessKey": "**********"
session=$(echo -- "$tokens" | sed -n 's!.*"SessionToken": "**********"
access=$(echo -- "$tokens" | sed -n 's!.*"AccessKeyId": "**********"
expire=$(echo -- "$tokens" | sed -n 's!.*"Expiration": "**********"

if [ -z "$secret" -o -z "$session" -o -z "$access" ]
then
  echo "Unable to get temporary credentials.  Could not find secret/access/session entries

$tokens" >&2
  exit 255
fi

echo 'Removing old mfa setting'
sed -i '' '/mfa/,$d' ~/.aws/credentials

echo 'Push new mfa token, key, id to credentials'
echo AWS_SESSION_TOKEN= "**********"
echo AWS_SECRET_ACCESS_KEY= "**********"
echo AWS_ACCESS_KEY_ID= "**********"

echo [mfa] >> ~/.aws/credentials
echo AWS_SESSION_TOKEN= "**********" >> ~/.aws/credentials
echo AWS_SECRET_ACCESS_KEY= "**********" >> ~/.aws/credentials
echo AWS_ACCESS_KEY_ID= "**********" >> ~/.aws/credentials

echo Keys valid until $expire >&2
redentials

echo Keys valid until $expire >&2
