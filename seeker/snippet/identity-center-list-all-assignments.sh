#date: 2023-10-27T17:06:20Z
#url: https://api.github.com/gists/256e5cb1a4cbc6d0fef5d2f4ff373629
#owner: https://api.github.com/users/atheiman

IDENTITY_CENTER_INSTANCE_ARN="$(aws sso-admin list-instances --output text --query 'Instances[0].InstanceArn')"
IDENTITY_STORE_ID="$(aws sso-admin list-instances --output text --query 'Instances[0].IdentityStoreId')"

for acctid in $(aws organizations list-accounts --query 'Accounts[][Id]' --output text); do
  echo "acct:$(aws organizations describe-account --account-id "$acctid" --output text --query 'Account.[Id, Email, Name]')"
  for psarn in $(aws sso-admin list-permission-sets-provisioned-to-account --account-id "$acctid" --instance-arn "$IDENTITY_CENTER_INSTANCE_ARN" --output text --query 'PermissionSets[]'); do
      echo "  permissionset:$(aws sso-admin describe-permission-set --instance-arn "$IDENTITY_CENTER_INSTANCE_ARN" --permission-set-arn "$psarn" --output text --query 'PermissionSet.[Name]')"
      for groupid in $(aws sso-admin list-account-assignments --account-id "$acctid" --instance-arn "$IDENTITY_CENTER_INSTANCE_ARN" --permission-set-arn "$psarn" --output text --query 'AccountAssignments[?PrincipalType==`GROUP`].[PrincipalId]'); do
        echo "    group:$(aws identitystore describe-group --identity-store-id "$IDENTITY_STORE_ID" --group-id "$groupid" --output text --query 'DisplayName')"
      done

      for userid in $(aws sso-admin list-account-assignments --account-id "$acctid" --instance-arn "$IDENTITY_CENTER_INSTANCE_ARN" --permission-set-arn "$psarn" --output text --query 'AccountAssignments[?PrincipalType==`USER`].[PrincipalId]'); do
        echo "    user:$(aws identitystore describe-user --identity-store-id "$IDENTITY_STORE_ID" --user-id "$userid" --output text --query 'UserName')"
      done
  done
  echo
done
