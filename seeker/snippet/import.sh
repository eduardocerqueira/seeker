#date: 2021-12-14T17:07:44Z
#url: https://api.github.com/gists/3dc0ab4b716eed5155a9732d81a3d210
#owner: https://api.github.com/users/janeczku

#!/usr/bin/env bash
set -xe

RANCHER_HOST="REDACTED.cloud"
BEARER_TOKEN="token-hcjwf:REDACTED"

if [[ $# -eq 0 ]] ; then
    echo "please specify cluster name"
    exit 1
fi

CLUSTER_NAME="$1"

if [[ -z "${KUBERNETES_SERVICE_HOST}" ]]; then
    echo "Running outside of a pod"
else
    if kubectl get namespace cattle-system; then
        echo "Cluster has already been registered"
        exit 0
    fi
fi

# Create cluster and extract cluster ID
RESP=`curl -ks "https://${RANCHER_HOST}/v3/cluster" -H 'content-type: application/json' -H "Authorization: Bearer $BEARER_TOKEN" --data-binary '{"type":"cluster","name":"'${CLUSTER_NAME}'","import":true}'`
CLUSTERID=`echo $RESP | jq -r .id`
echo "Cluster ID: ${CLUSTERID}"

# Generate registration token
ID=`curl -ks "https://${RANCHER_HOST}/v3/clusters/${CLUSTERID}/clusterregistrationtoken" -H 'content-type: application/json' -H "Authorization: Bearer $BEARER_TOKEN" --data-binary '{"type":"clusterRegistrationToken","clusterId":"'$CLUSTERID'"}' | jq -r .id`

sleep 2

# Extract Registration Command
COMMAND=`curl -ks "https://${RANCHER_HOST}/v3/clusters/${CLUSTERID}/clusterregistrationtoken/$ID" -H 'content-type: application/json' -H "Authorization: Bearer $BEARER_TOKEN" | jq -r .insecureCommand`
echo -p "Insecure Command: \n${COMMAND}"

# Uncomment to execute registration command
# eval "${COMMAND}"
