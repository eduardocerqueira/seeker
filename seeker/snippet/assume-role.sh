#date: 2022-11-09T17:12:53Z
#url: https://api.github.com/gists/53c23be0af43bb8fcfe2f2ba520b9b73
#owner: https://api.github.com/users/zaccharles

function ar() {
  export $(printf "AWS_ACCESS_KEY_ID= "**********"=%s AWS_SESSION_TOKEN=%s" \
  $(aws sts assume-role \
  --role-arn $1 \
  --role-session-name $USER \
  --query "Credentials.[AccessKeyId,SecretAccessKey,SessionToken]" \
  --output text))
}ut text))
}