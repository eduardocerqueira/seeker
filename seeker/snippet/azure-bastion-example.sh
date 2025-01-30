#date: 2025-01-30T17:10:44Z
#url: https://api.github.com/gists/c70995ebdff6852fb69cc7a70cb03ba2
#owner: https://api.github.com/users/dustinhennis

az network bastion tunnel \
  --name "BASTION_NAME" \
  --resource-group "RESOURCE_GROUP" \
  --target-resource-id "TARGET_NAME" \
  --resource-port "REMOTE_PORT" \
  --port "LOCAL_PORT"