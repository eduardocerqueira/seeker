#date: 2022-06-17T16:48:04Z
#url: https://api.github.com/gists/340ba9e0bef66f3b38b34fa02a89ce86
#owner: https://api.github.com/users/bsch150

#!/usr/bin/env bash

# This script is a demonstration of accurate AWS cli behavior when fetching secrets with stages and version IDs. 

SECRET_NAME="test-secret-delete-me"
AWS_PROFILE="some-aws-profile"

FIRST_VALUE="first_secret_value"
SECOND_VALUE="second_secret_value"

# create the secret
aws --profile consumption-dev-swat-admin \
  secretsmanager create-secret \
  --name ${SECRET_NAME}

# Put the first secret value as AWSCURRENT
first_put_response=$(aws --profile ${AWS_PROFILE} \
  secretsmanager put-secret-value \
  --secret-id ${SECRET_NAME} \
  --secret-string ${FIRST_VALUE} \
  --version-stages AWSCURRENT)
first_version=$(echo ${first_put_response} | jq -r .VersionId)

# This works fine
echo "Fetching version-id ${first_version} and stage 'AWSCURRENT'"
first_response=$(aws --profile ${AWS_PROFILE} \
  secretsmanager get-secret-value \
  --secret-id ${SECRET_NAME} \
  --version-stage AWSCURRENT \
  --version-id "${first_version}")

# Put the second value as AWSCURRENT
second_put_response=$(aws --profile ${AWS_PROFILE} \
  secretsmanager put-secret-value \
  --secret-id ${SECRET_NAME} \
  --secret-string ${SECOND_VALUE} \
  --version-stages "AWSCURRENT")
second_version=$(echo ${second_put_response} | jq -r .VersionId)

# Fetching new version with AWSCURRENT works fine
echo "Fetching version-id ${second_version} and stage 'AWSCURRENT'"
second_response=$(aws --profile ${AWS_PROFILE} \
  secretsmanager get-secret-value \
  --secret-id ${SECRET_NAME} \
  --version-stage "AWSCURRENT" \
  --version-id ${second_version})

# The first secret version is no longer AWSCURRENT. This breaks.
echo "Attempting to fetch version-id ${first_version} and stage 'AWSCURRENT'"
third_response=$(aws --profile ${AWS_PROFILE} \
  secretsmanager get-secret-value \
  --secret-id ${SECRET_NAME} \
  --version-stage "AWSCURRENT" \
  --version-id ${first_version})