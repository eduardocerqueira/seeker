#date: 2023-04-03T16:42:59Z
#url: https://api.github.com/gists/705c16539f53bee7d4cfd486e2ce3bd6
#owner: https://api.github.com/users/silug

#!/bin/bash

set -e

if [ "$#" -ne 1 ] ; then
    cat >&2 <<END_USAGE
Usage: $(basename "$0") <image_name>

Creates a kind cluster and deploys <image_name>.
END_USAGE
    exit 1
fi

image_name="$1"

CLUSTER_NAME=${CLUSTER_NAME:-loadtest}
MAX_PODS=${MAX_PODS:-2048}
CONTAINER_MEMORY_LIMIT=${CONTAINER_MEMORY_LIMIT:-30Mi}

cidr=$(awk -v num_hosts="$MAX_PODS" 'BEGIN{print int(32 - log(num_hosts + 2)/log(2))}')

sudo sysctl fs.inotify.max_user_watches=$(( MAX_PODS * 2048 ))
sudo sysctl fs.inotify.max_user_instances=$(( MAX_PODS * 2 ))
sudo sysctl net.ipv4.neigh.default.gc_thresh1=$(( MAX_PODS * 2 ))
sudo sysctl net.ipv4.neigh.default.gc_thresh2=$(( MAX_PODS * 8 ))
sudo sysctl net.ipv4.neigh.default.gc_thresh3=$(( MAX_PODS * 16 ))

cat > "kind-${CLUSTER_NAME}.yaml" <<END_KIND_YAML
---
kind: Cluster
apiVersion: kind.x-k8s.io/v1alpha4
networking:
  podSubnet: "10.0.0.0/$(( cidr - 2 ))"
nodes:
- role: control-plane
  kubeadmConfigPatches:
  - |
    kind: InitConfiguration
    nodeRegistration:
      kubeletExtraArgs:
        max-pods: "${MAX_PODS}"
  - |
    kind: ClusterConfiguration
    controllerManager:
      extraArgs:
        node-cidr-mask-size: "$(( cidr - 1 ))"
END_KIND_YAML

kind get clusters | grep -q "^${CLUSTER_NAME}$" \
    || kind create cluster -n "$CLUSTER_NAME" --config "kind-${CLUSTER_NAME}.yaml"
kubectl="kubectl --context kind-${CLUSTER_NAME}"

docker pull "$image_name"
kind load --name "$CLUSTER_NAME" docker-image "$image_name"

image_shortname="${image_name##*/}"
appname=scaleit

cat > deployment.yaml <<END_DEPLOYMENT_YAML
---
apiVersion: apps/v1
kind: Deployment
metadata:
  labels:
    app: ${appname}
  name: ${appname}
  namespace: default
spec:
  progressDeadlineSeconds: 600
  replicas: 1
  revisionHistoryLimit: 10
  selector:
    matchLabels:
      app: ${appname}
  strategy:
    rollingUpdate:
      maxSurge: 25%
      maxUnavailable: 25%
    type: RollingUpdate
  template:
    metadata:
      labels:
        app: ${appname}
    spec:
      containers:
        - image: ${image_name}
          imagePullPolicy: Never
          name: ${image_shortname%%:*}
          resources:
            requests:
              memory: ${CONTAINER_MEMORY_LIMIT}
            limits:
              memory: ${CONTAINER_MEMORY_LIMIT}
      dnsPolicy: ClusterFirst
status: {}
END_DEPLOYMENT_YAML

$kubectl apply -f deployment.yaml

cat >&2 <<END_MESSAGE

Scale the deployment with the following command:

  $kubectl scale deployment $appname --replicas <n>

where <n> is the number of pods to deploy.
Remove the cluster with the following command:

  kind delete cluster --name ${CLUSTER_NAME}

END_MESSAGE