#date: 2022-03-30T17:06:22Z
#url: https://api.github.com/gists/548d8f614f7ecabedb24bda6698c84e0
#owner: https://api.github.com/users/bhanu-prakashl

az login

az group create --name <resource-group-name> --location eastus
az appservice plan create --name <plan-name> --resource-group <resource-group-name> --is-linux
az webapp create --name <app-name> --plan <plan-name> --resource-group <resource-group-name> --runtime DOTNET:6.0

az webapp up --name <app-name> --plan <plan-name> --os-type linux --runtime DOTNET:6.0

az ad sp create-for-rbac --name <aad-app-name> --sdk-auth --role contributor --scopes /subscriptions/<subscription-id>/resourceGroups/<resource-group>