#date: 2024-01-17T16:40:41Z
#url: https://api.github.com/gists/2ba1d9e89a6b72414b0202b03b683890
#owner: https://api.github.com/users/SteveParson

#!/bin/bash

# Replace 'YourIAMRoleName' with the actual IAM role name
ROLE_NAME="YourIAMRoleName"

# List and get documents of attached policies
echo "Attached Policies:"
aws iam list-attached-role-policies --role-name "$ROLE_NAME" | jq -r '.AttachedPolicies[].PolicyArn' | while read -r policy_arn; do
    policy_version_id=$(aws iam get-policy --policy-arn "$policy_arn" | jq -r '.Policy.DefaultVersionId')
    policy_document=$(aws iam get-policy-version --policy-arn "$policy_arn" --version-id "$policy_version_id" | jq -r '.PolicyVersion.Document')
    echo "Policy ARN: $policy_arn"
    echo "Policy Document: $policy_document"
done

# List and get documents of inline policies
echo "Inline Policies:"
aws iam list-role-policies --role-name "$ROLE_NAME" | jq -r '.PolicyNames[]' | while read -r policy_name; do
    policy_document=$(aws iam get-role-policy --role-name "$ROLE_NAME" --policy-name "$policy_name" | jq -r '.PolicyDocument')
    echo "Policy Name: $policy_name"
    echo "Policy Document: $policy_document"
done
