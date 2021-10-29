#date: 2021-10-29T17:05:16Z
#url: https://api.github.com/gists/9500b074840eea2197dfb85cd294da68
#owner: https://api.github.com/users/dgvigil

#!/bin/bash
# Needs to be run in embedded kubectl of Imported cluster called local, or using the RKE generated kubeconfig

# Check if jq exists
command -v jq >/dev/null 2>&1 || { echo "jq is not installed. Exiting." >&2; exit 1; }

# Check if clustername is given
if [ -z "$1" ]; then
    echo "Usage: $0 [clustername]"
    exit 1
fi

# Provide clustername as first argument
CLUSTERNAME=$1

# Validate that we are querying the correct etcd
if kubectl get clusters.management.cattle.io > /dev/null; then
  echo "'kubectl get cluster' returns clusters available"
else
  echo "'kubectl get cluster' returns error, this should be run in the embedded kubectl of the 'local' imported cluster"                                                                                                                                                                   
  exit 1
fi

# Get clusters
CLUSTERINFO=$(kubectl get clusters.management.cattle.io --no-headers --output=custom-columns=Name:.spec.displayName,Driver:.status.driver,ClusterID:.metadata.name)
echo "Clusters found:"
echo "${CLUSTERINFO}"

# Get clusterid from clustername
CLUSTERID=$(echo "${CLUSTERINFO}" | awk -v CLUSTERNAME=$CLUSTERNAME '$1==CLUSTERNAME { print $3 }')

if [[ -z $CLUSTERID ]]; then
  echo "No CLUSTERID could be retrieved for $CLUSTERNAME, make sure you entered the correct clustername"
  exit 1
fi

CLUSTERDRIVER=$(echo "${CLUSTERINFO}" | awk -v CLUSTERNAME=$CLUSTERNAME '$1==CLUSTERNAME { print $2 }')

if [[ $CLUSTERDRIVER != "rancherKubernetesEngine" ]]; then
  echo "Cluster ${CLUSTERNAME} is not a RKE built cluster, you can't retrieve the kubeconfig with this script"
  echo "If you have built your cluster using RKE and then imported, the 'kube_config_cluster.yml' file can be used"
  echo "If you have imported your own cluster, the kubeconfig was created when building your cluster"
  exit 1
fi

# Get kubeconfig for cluster ID and save it to `kubeconfig`
if kubectl get secret c-$CLUSTERID -n cattle-system; then
  echo "Secret c-${CLUSTERID} found"
  kubectl get secret c-$CLUSTERID -n cattle-system -o json | jq -r .data.cluster | base64 -d | jq -r .metadata.state > kubeconfig
  KUBECONTENT=$(cat kubeconfig)
  if [[ "${KUBECONTENT}" != "null" ]]; then
    echo "Kubeconfig written to file 'kubeconfig'"
  else
    echo "Kubeconfig could not be retrieved"
  fi
else
  echo "Secret c-${CLUSTERID} could not be found"
  echo "The cluster is not provisioned yet or is not in healthy state"
  exit 1
fi