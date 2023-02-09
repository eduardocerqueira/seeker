#date: 2023-02-09T16:55:05Z
#url: https://api.github.com/gists/f08505faed6a5742a1ec5f1a9c5d489c
#owner: https://api.github.com/users/sumansrivastava

#!/bin/bash

# Prompt for the namespace
echo "Enter the namespace: "
read namespace

# Check all the resources (pods, Services, Deployment, events)
resources=(pods services deployments events)

for resource in "${resources[@]}"
do
  echo "Checking $resource in namespace $namespace"
  kubectl get $resource -n $namespace > "$resource.txt"

  # Collect only report from the errors
  error_count=$(kubectl describe $resource -n $namespace | grep -i "error" | wc -l)
  if [ $error_count -gt 0 ]
  then
    kubectl describe $resource -n $namespace | grep -i "error" > "$resource-errors.txt"
  else
    echo "No errors found in $resource"
  fi

  # Save all the logs in the same directory with the resource names
  kubectl logs $(kubectl get pods -n $namespace | awk '{print $1}' | grep -v NAME) -n $namespace > "$resource-logs.txt"
done