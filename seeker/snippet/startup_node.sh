#date: 2025-10-21T17:14:05Z
#url: https://api.github.com/gists/71367642611902ca93449d4e1e9aa148
#owner: https://api.github.com/users/dingp

#!/bin/bash

# This script restores the replica counts of Deployments and StatefulSets
# that were previously scaled down by the shutdown_node.sh script,
# optionally filtering by the node name where they were originally running.
#
# Usage: startup_node.sh [<node_name>] [--dry-run]

set -euo pipefail

DRY_RUN=false
NODE_FILTER=""

# Function to get current timestamp in PDT
get_pdt_timestamp() {
  date +"%Y-%m-%d %H:%M:%S PDT"
}

# Parse arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=true
      shift
      ;;
    *)
      if [[ -z "$NODE_FILTER" ]]; then
        NODE_FILTER="$1"
      else
        echo "Error: Too many arguments. Usage: $0 [<node_name>] [--dry-run]"
        exit 1
      fi
      shift
      ;;
  esac
done

if [[ "$DRY_RUN" == "true" ]]; then
  echo "--- DRY RUN MODE ---"
fi

if [[ -n "$NODE_FILTER" ]]; then
  echo "Finding resources to scale up for node: ${NODE_FILTER}...\n"
else
  echo "Finding all resources to scale up...\n"
fi

printf "% -15s % -15s % -30s % -30s % -10s % -25s % -25s\n" "ACTION" "TYPE" "NAME" "NAMESPACE" "REPLICAS" "ORIGINAL NODE" "TIMESTAMP"
printf "% -15s % -15s % -30s % -30s % -10s % -25s % -25s\n" "------" "----" "----" "---------" "--------" "-------------" "---------"

# Iterate through both deployments and statefulsets
for RESOURCE_TYPE in deployment statefulset; do
  # Get all resources of the current type that have the 'original-replicas' annotation
  RESOURCES=$(kubectl get "$RESOURCE_TYPE" --all-namespaces -o json | jq -r '.items[] | select(.metadata.annotations."original-replicas") | .metadata.name + "/" + .metadata.namespace')

  for RESOURCE_INFO in $RESOURCES; do
    CURRENT_TIMESTAMP=$(get_pdt_timestamp)
    RESOURCE_NAME=$(echo "$RESOURCE_INFO" | cut -d'/' -f1)
    RESOURCE_NAMESPACE=$(echo "$RESOURCE_INFO" | cut -d'/' -f2)

    REPLICAS=$(kubectl get "$RESOURCE_TYPE" "$RESOURCE_NAME" --namespace "$RESOURCE_NAMESPACE" -o jsonpath='{.metadata.annotations.original-replicas}')
    ORIGINAL_NODE=$(kubectl get "$RESOURCE_TYPE" "$RESOURCE_NAME" --namespace "$RESOURCE_NAMESPACE" -o jsonpath='{.metadata.annotations.original-node}')

    if [[ -n "$NODE_FILTER" && "$ORIGINAL_NODE" != "$NODE_FILTER" ]]; then
      continue
    fi

    if [[ -n "$REPLICAS" ]]; then
      if [[ "$DRY_RUN" == "true" ]]; then
        printf "% -15s % -15s % -30s % -30s % -10s % -25s % -25s\n" "DRY RUN" "$RESOURCE_TYPE" "$RESOURCE_NAME" "$RESOURCE_NAMESPACE" "0 -> $REPLICAS" "$ORIGINAL_NODE" "$CURRENT_TIMESTAMP"
      else
        kubectl scale "$RESOURCE_TYPE" "$RESOURCE_NAME" --namespace "$RESOURCE_NAMESPACE" --replicas="$REPLICAS"
        kubectl annotate "$RESOURCE_TYPE" "$RESOURCE_NAME" --namespace "$RESOURCE_NAMESPACE" original-replicas- original-node-
        printf "% -15s % -15s % -30s % -30s % -10s % -25s % -25s\n" "SCALED UP" "$RESOURCE_TYPE" "$RESOURCE_NAME" "$RESOURCE_NAMESPACE" "0 -> $REPLICAS" "$ORIGINAL_NODE" "$CURRENT_TIMESTAMP"
      fi
    fi
  done
done

echo "\nAll specified workloads are being restored."
if [[ "$DRY_RUN" == "true" ]]; then
  echo "--- DRY RUN COMPLETE --- No changes were made."
fi