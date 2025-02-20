#date: 2025-02-20T16:46:32Z
#url: https://api.github.com/gists/c9a0b1cd5260e335f25843c034e3f57c
#owner: https://api.github.com/users/lfarci

#!/bin/bash

# ---------------------------------------------------------------------------- #
#                                   FUNCTIONS                                  #
# ---------------------------------------------------------------------------- #

display_message() {
  local message_type=$1
  local message=$2

  case $message_type in
    "error")
      echo -e "\e[31mERROR: $message\e[0m"
      ;;
    "success")
      echo -e "\e[32m$message\e[0m"
      ;;
    "warning")
      echo -e "\e[33mWARNING: $message\e[0m"
      ;;
    "info")
    echo "INFO: $message"
      ;;
    "progress")
      echo -e "\e[34m$message\e[0m" # Blue for progress
      ;;
    *)
      echo "$message"
      ;;
  esac
}

display_progress() {
  local message=$1
  display_message progress "$message"
}

display_blank_line() {
  echo ""
}

get_short_location() {
    # Read JSON with all the locations
    local locations=$(cat ./common-modules/naming/locations.json)
    # Get the short location where the input is the key and the short location is the value
    local short_location=$(echo $locations | jq -r ".$1")
    echo $short_location
}

# ---------------------------------------------------------------------------- #
#                             INTRODUCTION MESSAGE                             #
# ---------------------------------------------------------------------------- #

display_blank_line
display_progress "Deploying the Secure Baseline scenario for Azure Red Hat Openshift"

# ---------------------------------------------------------------------------- #
#                                  PARAMETERS                                  #
# ---------------------------------------------------------------------------- #

HUB_WORKLOAD_NAME=${HUB_WORKLOAD_NAME:-"hub"}
SPOKE_WORKLOAD_NAME=${SPOKE_WORKLOAD_NAME:-"aro-lza"}
ENVIRONMENT=${ENVIRONMENT:-"DEV"}
LOCATION=${LOCATION:-"eastus"}

_environment_lower_case=$(echo $ENVIRONMENT | tr '[:upper:]' '[:lower:]')
_short_location=$(get_short_location $LOCATION)

display_message info "Hub workload name: $HUB_WORKLOAD_NAME"
display_message info "Spoke workload name: $SPOKE_WORKLOAD_NAME"
display_message info "Environment: $ENVIRONMENT"
display_message info "Location: $LOCATION"
if [ -z "$HASH" ]; then
    HASH_WITH_HYPHEN=""
    display_message info "Hash: not using hash"
else
    HASH_WITH_HYPHEN=-$HASH
    display_message info "Hash: $HASH"
fi
display_blank_line

# ---------------------------------------------------------------------------- #
#                              REGISTRER PROVIDERS                             #
# ---------------------------------------------------------------------------- #

display_progress "Registering providers"
az provider register --namespace 'Microsoft.RedHatOpenShift' --wait
az provider register --namespace 'Microsoft.Compute' --wait
az provider register --namespace 'Microsoft.Storage' --wait
az provider register --namespace 'Microsoft.Authorization' --wait

display_progress "Enable encryption at host"
az feature registration create --name EncryptionAtHost --namespace Microsoft.Compute
display_progress "Registration of providers completed successfully"
display_blank_line

# ---------------------------------------------------------------------------- #
#                                      HUB                                     #
# ---------------------------------------------------------------------------- #

# Deploy the hub resources
_hub_deployment_name="$HUB_WORKLOAD_NAME-$_environment_lower_case-$_short_location$HASH_WITH_HYPHEN"
display_progress "Deploying the hub resources"
display_message info "Deployment name: $_hub_deployment_name"
if [ -z "$HASH" ]; then
    az deployment sub create \
        --name $_hub_deployment_name \
        --location $LOCATION \
        --template-file "./01-Hub/main.bicep" \
        --parameters ./01-Hub/main.bicepparam \
        --parameters \
            workloadName=$HUB_WORKLOAD_NAME \
            env=$ENVIRONMENT \
            location=$LOCATION
else
    az deployment sub create \
        --name $_hub_deployment_name \
        --location $LOCATION \
        --template-file "./01-Hub/main.bicep" \
        --parameters ./01-Hub/main.bicepparam \
        --parameters \
            hash=$HASH \
            workloadName=$HUB_WORKLOAD_NAME \
            env=$ENVIRONMENT \
            location=$LOCATION
fi

# Get the outputs from the hub deployment
HUB_RG_NAME=$(az deployment sub show --name "$_hub_deployment_name" --query "properties.outputs.resourceGroupName.value" -o tsv)
HUB_VNET_ID=$(az deployment sub show --name "$_hub_deployment_name" --query "properties.outputs.virtualNetworkResourceId.value" -o tsv)
LOG_ANALYTICS_WORKSPACE_ID=$(az deployment sub show --name "$_hub_deployment_name" --query "properties.outputs.logAnalyticsWorkspaceResourceId.value" -o tsv)
KEY_VAULT_PRIVATE_DNS_ZONE_RESOURCE_ID=$(az deployment sub show --name "$_hub_deployment_name" --query "properties.outputs.keyVaultPrivateDnsZoneResourceId.value" -o tsv)
KEY_VAULT_PRIVATE_DNS_ZONE_NAME=$(az deployment sub show --name "$_hub_deployment_name" --query "properties.outputs.keyVaultPrivateDnsZoneName.value" -o tsv)
ACR_PRIVATE_DNS_ZONE_RESOURCE_ID=$(az deployment sub show --name "$_hub_deployment_name" --query "properties.outputs.acrPrivateDnsZoneResourceId.value" -o tsv)
ACR_PRIVATE_DNS_ZONE_NAME=$(az deployment sub show --name "$_hub_deployment_name" --query "properties.outputs.acrPrivateDnsZoneName.value" -o tsv)
FIREWALL_PRIVATE_IP=$(az deployment sub show --name "$_hub_deployment_name" --query "properties.outputs.firewallPrivateIp.value" -o tsv)
display_progress "Hub resources deployed successfully"
display_blank_line