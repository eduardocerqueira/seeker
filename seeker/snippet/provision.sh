#date: 2025-04-11T16:46:41Z
#url: https://api.github.com/gists/7e842929d864593c08640433a7407885
#owner: https://api.github.com/users/ConnorBaker

#!/usr/bin/env bash

# A wrapper around main.tf

set -euo pipefail

declare -r AZURE_INSTANCE_USER="azure"
declare -r AZURE_INSTANCE_SIZE="Standard_HB120rs_v3"
declare -r AZURE_SSH_PUBLIC_KEY_PATH="$HOME/.ssh/azure_ed25519.pub"

azureSetup() {
    az account set --subscription "Azure subscription 1"
    az configure --defaults group=simpleLinuxTestVMResourceGroup location=eastus
    az group create --resource-group simpleLinuxTestVMResourceGroup --location eastus
}

provisionInstances() {
    az deployment group create \
        --name simpleLinuxTestVMDeployment \
        --template-file main.bicep \
        --parameters \
            adminUsername="$AZURE_INSTANCE_USER" \
            sshPublicKey=@"$AZURE_SSH_PUBLIC_KEY_PATH" \
            vmSize="$AZURE_INSTANCE_SIZE" \
        --output json \
        --verbose |
        jq -crS .
}

main() {
    azureSetup
    provisionInstances
}

main
