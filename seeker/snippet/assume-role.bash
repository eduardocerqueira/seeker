#date: 2024-04-11T17:06:50Z
#url: https://api.github.com/gists/babf83ef7d638e038deda7be28ba40ab
#owner: https://api.github.com/users/gene1wood

#!/bin/bash -x

# Session duration
# DURATION=43200
DURATION=3600

unset AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_SESSION_TOKEN

if ! sts=( $(
  aws sts assume-role \
  --duration-seconds "$DURATION" \
  --role-arn $1 \
  --role-session-name gene \
  --query 'Credentials.[AccessKeyId,SecretAccessKey,SessionToken]' \
  --output text
) ); then
  exit 1
fi

echo "export AWS_ACCESS_KEY_ID= "**********"=${sts[1]} AWS_SESSION_TOKEN=${sts[2]}"
_TOKEN=${sts[2]}"
