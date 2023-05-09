#date: 2023-05-09T17:07:33Z
#url: https://api.github.com/gists/042fc30cb68ba0b4f24896df124b280a
#owner: https://api.github.com/users/raksit31667

az config set extension.use_dynamic_install=yes_without_prompt # Install extension ให้อัตโนมัติ

provisioningState=""
while [[ "$provisioningState" == "Provisioning" ]]; do
  echo "Waiting for provisioning..."
  sleep 10
  echo "Checking provisioning state..."

  # แก้บรรทัดข้างล่าง
  provisioningState=$(az containerapp revision list -n "<your-aca-name>" -g "<your-resource-group>" --query "[?contains(properties.template.containers[0].image, '$BUILD_NUMBER')].properties.provisioningState" -o tsv)
done
if [[ "$provisioningState" == "Provisioned" ]]; then
  echo "ACA deployment successful"
  exit 0
else
  echo "ACA deployment failed"
  exit 1
fi