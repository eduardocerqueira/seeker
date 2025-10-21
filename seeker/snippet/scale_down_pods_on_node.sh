#date: 2025-10-21T17:14:05Z
#url: https://api.github.com/gists/71367642611902ca93449d4e1e9aa148
#owner: https://api.github.com/users/dingp

#!/bin/bash

# This script scales down all deployments and statefulsets running on a given node to 0 replicas.
# It stores the original replica count and the node name in annotations for later restoration.
#
# Usage: scale_down_pods_on_node.sh <node_name> [--dry-run]

set -euo pipefail

DRY_RUN=false
NODE_NAME=""

# Namespaces to skip during shutdown
SKIP_NAMESPACES=(
  cattle-dashboards
  cattle-fleet-system
  cattle-gatekeeper-system
  cattle-impersonation-system
  cattle-logging-system
  cattle-monitoring-system
  cattle-neuvector-system
  cattle-prometheus
  cattle-system
  cis-operator-system
  external-dns
  fleet-system
  ingress-nginx
  kube-node-lease
  kube-public
  kube-system
  longhorn-system
  metallb
  nersc-webhooks
  nfs-client-provisioner
  nfs-subdir-external-provisioner
  opa
  pathchecker
  security-scan
)

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
      if [[ -z "$NODE_NAME" ]]; then
        NODE_NAME="$1"
      else
        echo "Error: Too many arguments. Usage: $0 <node_name> [--dry-run]"
        exit 1
      fi
      shift
      ;; 
  esac
done

if [[ -z "$NODE_NAME" ]]; then
  echo "Usage: $0 <node_name> [--dry-run]"
  exit 1
fi

if [[ "$DRY_RUN" == "true" ]]; then
  echo "--- DRY RUN MODE ---"
fi

# Convert SKIP_NAMESPACES array to a JSON array for jq
SKIP_NAMESPACES_JSON=$(printf '%s\n' "${SKIP_NAMESPACES[@]}" | jq -R . | jq -s .)

# Get all pods on the specified node, then get their unique owners (Deployments or StatefulSets).
# Filter out pods from skipped namespaces.
# A temporary file is used to store the list of owners.
OWNERS_FILE=$(mktemp)
trap 'rm -f -- "$OWNERS_FILE"' EXIT

kubectl get pods --all-namespaces --field-selector "spec.nodeName=${NODE_NAME}" -o json |
  jq -r --argjson skip_namespaces "$SKIP_NAMESPACES_JSON"  \
  '.items[] | {owner: .metadata.ownerReferences[]?, namespace: .metadata.namespace} | select(.owner.kind=="ReplicaSet" or .owner.kind=="StatefulSet") | select([.namespace] | inside($skip_namespaces) | not) | .owner.kind + "/" + .owner.name + "/" + .namespace' | 
  sort -u >"$OWNERS_FILE"

if ! [[ -s "$OWNERS_FILE" ]]; then
  echo "No Deployments or StatefulSets found managing pods on node ${NODE_NAME}"
  exit 0
fi

printf "% -15s % -15s % -30s % -30s % -10s % -25s % -25s\n" "ACTION" "TYPE" "NAME" "NAMESPACE" "REPLICAS" "NODENAME" "TIMESTAMP"
printf "% -15s % -15s % -30s % -30s % -10s % -25s % -25s\n" "------" "----" "----" "---------" "--------" "--------" "---------"

while IFS= read -r OWNER_INFO; do
  CURRENT_TIMESTAMP=$(get_pdt_timestamp)
  OWNER_KIND=$(echo "$OWNER_INFO" | cut -d'/' -f1)
  OWNER_NAME=$(echo "$OWNER_INFO" | cut -d'/' -f2)
  OWNER_NAMESPACE=$(echo "$OWNER_INFO" | cut -d'/' -f3) # Extract namespace

  RESOURCE_TYPE=""
  RESOURCE_NAME=""
  RESOURCE_NAMESPACE=$OWNER_NAMESPACE # Use the extracted namespace

  if [[ "$OWNER_KIND" == "ReplicaSet" ]]; then
    # Check if the ReplicaSet exists and has an owner (i.e., a Deployment)
    REPLICASET_JSON=$(kubectl get replicaset "$OWNER_NAME" --namespace "$OWNER_NAMESPACE" -o json 2>/dev/null || true)
    if [[ -z "$REPLICASET_JSON" ]]; then
      printf "% -15s % -15s % -30s % -30s % -10s % -25s % -25s\n" "SKIPPING" "ReplicaSet" "$OWNER_NAME" "$OWNER_NAMESPACE" "N/A (Not Found)" "$NODE_NAME" "$CURRENT_TIMESTAMP"
      continue
    fi

    HAS_OWNER=$(echo "$REPLICASET_JSON" | jq -e '.metadata.ownerReferences != null' >/dev/null && echo "true" || echo "false")

    if [[ "$HAS_OWNER" == "true" ]]; then
      DEPLOYMENT_NAME=$(echo "$REPLICASET_JSON" | jq -r '.metadata.ownerReferences[0].name')
      if [[ -n "$DEPLOYMENT_NAME" ]]; then
        RESOURCE_TYPE="deployment"
        RESOURCE_NAME=$DEPLOYMENT_NAME
      else
        printf "% -15s % -15s % -30s % -30s % -10s % -25s % -25s\n" "SKIPPING" "ReplicaSet" "$OWNER_NAME" "$OWNER_NAMESPACE" "N/A (No Deployment Owner)" "$NODE_NAME" "$CURRENT_TIMESTAMP"
        continue
      fi
    else
      printf "% -15s % -15s % -30s % -30s % -10s % -25s % -25s\n" "SKIPPING" "ReplicaSet" "$OWNER_NAME" "$OWNER_NAMESPACE" "N/A (Not Owned by Deployment)" "$NODE_NAME" "$CURRENT_TIMESTAMP"
      continue
    fi
  elif [[ "$OWNER_KIND" == "StatefulSet" ]]; then
    RESOURCE_TYPE="statefulset"
    RESOURCE_NAME=$OWNER_NAME
  else
    printf "% -15s % -15s % -30s % -30s % -10s % -25s % -25s\n" "SKIPPING" "$OWNER_KIND" "$OWNER_NAME" "$OWNER_NAMESPACE" "N/A (Unsupported Kind)" "$NODE_NAME" "$CURRENT_TIMESTAMP"
    continue
  fi

  # Get the current number of replicas
  REPLICAS=$(kubectl get "$RESOURCE_TYPE" "$RESOURCE_NAME" --namespace "$RESOURCE_NAMESPACE" -o jsonpath='{.spec.replicas}')

  # Check if the annotation already exists to avoid overwriting it
  ANNOTATION_REPLICAS=$(kubectl get "$RESOURCE_TYPE" "$RESOURCE_NAME" --namespace "$RESOURCE_NAMESPACE" -o jsonpath='{.metadata.annotations.original-replicas}')
  ANNOTATION_NODE=$(kubectl get "$RESOURCE_TYPE" "$RESOURCE_NAME" --namespace "$RESOURCE_NAMESPACE" -o jsonpath='{.metadata.annotations.original-node}')

  if [[ -n "$ANNOTATION_REPLICAS" && -n "$ANNOTATION_NODE" ]]; then
    printf "% -15s % -15s % -30s % -30s % -10s % -25s % -25s\n" "SKIPPING" "$RESOURCE_TYPE" "$RESOURCE_NAME" "$RESOURCE_NAMESPACE" "$REPLICAS (Already Scaled Down)" "$NODE_NAME" "$CURRENT_TIMESTAMP"
    continue
  fi

  if [[ "$DRY_RUN" == "true" ]]; then
    printf "% -15s % -15s % -30s % -30s % -10s % -25s % -25s\n" "DRY RUN" "$RESOURCE_TYPE" "$RESOURCE_NAME" "$RESOURCE_NAMESPACE" "$REPLICAS -> 0" "$NODE_NAME" "$CURRENT_TIMESTAMP"
  else
    # Store the original number of replicas and the node name in annotations
    kubectl annotate "$RESOURCE_TYPE" "$RESOURCE_NAME" --namespace "$RESOURCE_NAMESPACE" "original-replicas=${REPLICAS}" "original-node=${NODE_NAME}" --overwrite
    # Scale the resource down to 0 replicas
    kubectl scale "$RESOURCE_TYPE" "$RESOURCE_NAME" --namespace "$RESOURCE_NAMESPACE" --replicas=0
    printf "% -15s % -15s % -30s % -30s % -10s % -25s % -25s\n" "SCALED DOWN" "$RESOURCE_TYPE" "$RESOURCE_NAME" "$RESOURCE_NAMESPACE" "$REPLICAS -> 0" "$NODE_NAME" "$CURRENT_TIMESTAMP"
  fi
done <"$OWNERS_FILE"

echo "\nAll workloads on node ${NODE_NAME} are being processed."
if [[ "$DRY_RUN" == "true" ]]; then
  echo "--- DRY RUN COMPLETE --- No changes were made."
fi