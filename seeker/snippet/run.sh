#date: 2022-02-18T16:49:04Z
#url: https://api.github.com/gists/95f5442c28bea8b9d98302361ee485d3
#owner: https://api.github.com/users/danielscholl

#!/bin/bash

# Random string generator - don't change this.
RAND="$(echo $RANDOM | tr '[0-9]' '[a-z]')"

LOCATION="eastus"
RESOURCEGROUP="container-deployment-$RAND"

if [ -z $1 ]; then
  tput setaf 1; echo 'ERROR: URL Location of Deployment Script not provided' ; tput sgr0
  usage;
fi


# Get commandline for Azure CLI
az=$(which az)

# Fetch the CloudShell subscription ID
subId=$($az account show --query id -o tsv 2>/dev/null)

echo "==============================================================================================================================================================="
if [ ! "$($az group show -n $RESOURCEGROUP --query tags.currentStatus -o tsv 2>/dev/null)" = "groupCreated" ]; then
    # Deploy the resource group and update Status Tag
    echo "Deploying the resource group."
    $az group create -g "$RESOURCEGROUP" -l "$LOCATION" -o none 2>/dev/null
    $az group update -n $RESOURCEGROUP --tag currentStatus=groupCreated 2>/dev/null
    echo "done."
fi

echo "==============================================================================================================================================================="

if [ ! "$($az group show -n $RESOURCEGROUP --query tags.currentStatus -o tsv 2>/dev/null)" = "containerCreated" ]; then
    echo "Deploying the container (might take 2-3 minutes)..."
    $az container create -g $RESOURCEGROUP --name containerdeploy --image danielscholl/container-deploy --restart-policy Never --environment-variables subId=$subId RAND=$RAND SETUP_SCRIPT=$1 -o none 2>/dev/null
    $az group update -n $RESOURCEGROUP --tag currentStatus=containerCreated 2>/dev/null
    echo "done."
fi

echo "==============================================================================================================================================================="
echo "==============================================================================================================================================================="
echo "If cloudshell times out copy this command and run it again when cloud shell is restarted:"
echo "     az container logs --follow -n containerDeploy -g $RESOURCEGROUP"
echo "==============================================================================================================================================================="
echo "==============================================================================================================================================================="

if [ "$($az group show -n $RESOURCEGROUP --query tags.currentStatus -o tsv 2>/dev/null)" = "containerCreated" ]; then
    echo "Tail Logs"
    $az container logs -n containerDeploy -g $RESOURCEGROUP 2>/dev/null
fi
