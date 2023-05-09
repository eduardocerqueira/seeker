#date: 2023-05-09T17:07:04Z
#url: https://api.github.com/gists/1507f52b26824ce06fd063ac268870da
#owner: https://api.github.com/users/raksit31667

az config set extension.use_dynamic_install=yes_without_prompt # Install extension ให้อัตโนมัติ

provisioningState=""
while [[ "$provisioningState" == "Provisioning" ]]; do
  echo "Waiting for provisioning..."
  sleep 10
  echo "Checking provisioning state..."
  provisioningState=$(az containerapp revision list -n "<your-aca-name>" -g "<your-resource-group>" --query ".properties provisioningState" -o tsv)
done
if [[ "$provisioningState" == "Provisioned" ]]; then
  echo "ACA deployment successful"
  exit 0
else
  echo "ACA deployment failed"
  exit 1
fi