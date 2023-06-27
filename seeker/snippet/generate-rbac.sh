#date: 2023-06-27T17:07:35Z
#url: https://api.github.com/gists/e20d431a6b0572c14c68a2447bc2bc34
#owner: https://api.github.com/users/joelanford

#!/usr/bin/env bash

set -o errexit
set -o nounset
set -o pipefail

declare -a TMPROOT
declare -a CLUSTER_NAME

# We're going to do file manipulation, so let's work in a temp dir
TMPROOT="$(mktemp -p . -d 2>/dev/null || mktemp -d ./tmp-generate-rbac-XXXXXXX)"
# Make sure to delete the temp dir when we exit
#trap 'rm -rf $TMPROOT' EXIT

CLUSTER_NAME=$(basename "${TMPROOT}" | tr '[:upper:]' '[:lower:]')

pushd "${TMPROOT}"


######
# Create a kind cluster with auditing enabled
#####
cat <<EOF > audit-policy.yaml
apiVersion: audit.k8s.io/v1
kind: Policy
rules:
- level: Metadata
EOF

cat <<EOF > kind-config.yaml
kind: Cluster
apiVersion: kind.x-k8s.io/v1alpha4
nodes:
- role: control-plane
  kubeadmConfigPatches:
  - |
    kind: ClusterConfiguration
    apiServer:
        # enable auditing flags on the API server
        extraArgs:
          audit-log-path: /var/log/kubernetes/kube-apiserver-audit.log
          audit-policy-file: /etc/kubernetes/policies/audit-policy.yaml
        # mount new files / directories on the control plane
        extraVolumes:
          - name: audit-policies
            hostPath: /etc/kubernetes/policies
            mountPath: /etc/kubernetes/policies
            readOnly: true
            pathType: "DirectoryOrCreate"
          - name: "audit-logs"
            hostPath: "/var/log/kubernetes"
            mountPath: "/var/log/kubernetes"
            readOnly: false
            pathType: DirectoryOrCreate
  # mount the local file on the control plane
  extraMounts:
  - hostPath: ./audit-policy.yaml
    containerPath: /etc/kubernetes/policies/audit-policy.yaml
    readOnly: true
EOF

kind create cluster --config kind-config.yaml --name "${CLUSTER_NAME}"
#trap "kind delete cluster --name ${CLUSTER_NAME}" EXIT


######
# Apply the manifests to the cluster. Also add an extra cluster-admin binding to the cluster-olm-operator
# so that it can do everything. This avoid short-circuit scenarios.
######
kubectl apply -f ../vendor/github.com/openshift/api/config/v1/0000_00_cluster-version-operator_01_clusteroperator.crd.yaml
kubectl apply -f ../vendor/github.com/openshift/api/operator/v1alpha1/0000_10_config-operator_01_olm.crd.yaml
#kubectl create clusterrolebinding cluster-olm-operator-admin --clusterrole=cluster-admin --user=system:serviceaccount:openshift-cluster-olm-operator:cluster-olm-operator
kubectl apply -f ../manifests

######
# Exercise the operator
#  - Wait until it fully reconciles the OLM "cluster" object
#  - Delete the OLM "cluster" object and wait until it handles any finalizers
######
#kubectl wait clusteroperators.config.openshift.io/olm --for condition=Available --timeout=60s
#kubectl delete olms.operator.openshift.io cluster
#kubectl wait olms.operator.openshift.io/cluster --for=delete

######
# Generate the RBAC
######
#docker exec tmp.xl4losmkz4-control-plane cat /var/log/kubernetes/kube-apiserver-audit.log | audit2rbac --user system:serviceaccount:openshift-cluster-olm-operator:cluster-olm-operator -f - > rbac.yaml
