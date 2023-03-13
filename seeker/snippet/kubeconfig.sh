#date: 2023-03-13T16:44:18Z
#url: https://api.github.com/gists/ace184979cdb8f6533a8f10ff5f190c9
#owner: https://api.github.com/users/ChrisTomAlx

#!/bin/bash

# Copyright 2020 Gravitational, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This script creates a new k8s Service Account and generates a kubeconfig with
# its credentials. This Service Account has all the necessary permissions. 
# The kubeconfig is written in the current directory.
#
# You must configure your local kubectl to point to the right k8s cluster and
# have admin-level access.
#
# Note: all of the k8s resources are created in namespace "teleport".
#
# You can override the default namespace "teleport" using the
# TELEPORT_NAMESPACE environment variable.
# You can override the default service account name "teleport-sa" using the
# TELEPORT_SA_NAME environment variable.

set -eu -o pipefail

# Allow passing in common name and username in environment. If not provided,
# use default.
TELEPORT_SA=${TELEPORT_SA_NAME:-teleport-sa}
NAMESPACE=${TELEPORT_NAMESPACE:-teleport}

# Set OS specific values.
if [[ "$OSTYPE" == "linux-gnu" ]]; then
    BASE64_DECODE_FLAG="-d"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    BASE64_DECODE_FLAG="-D"
elif [[ "$OSTYPE" == "linux-musl" ]]; then
    BASE64_DECODE_FLAG="-d"
else
    echo "Unknown OS ${OSTYPE}"
    exit 1
fi

echo "Creating the Kubernetes Service Account with minimal RBAC permissions."
kubectl apply -f - <<EOF
apiVersion: v1
kind: Namespace
metadata:
  name: ${NAMESPACE}
---
apiVersion: v1
kind: ServiceAccount
metadata:
  name: ${TELEPORT_SA}
  namespace: ${NAMESPACE}
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: teleport-role
rules:
- apiGroups: ['*']
  resources: ['*']
  verbs: ['*']
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRoleBinding
metadata:
  name: teleport-crb
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: ClusterRole
  name: teleport-role
subjects:
- kind: ServiceAccount
  name: ${TELEPORT_SA}
  namespace: ${NAMESPACE}
EOF

# Checks if secret entry was defined for Service account. If defined it means that Kubernetes server has a
# version bellow 1.24, otherwise one must manually create the secret and bind it to the Service account to have a non expiring token.
# After Kubernetes v1.24 Service accounts no longer generate automatic tokens/secrets.
# We can use kubectl create token but the token has a expiration time.
# https://github.com/kubernetes/kubernetes/blob/master/CHANGELOG/CHANGELOG-1.24.md#urgent-upgrade-notes
SA_SECRET_NAME= "**********"={.secrets[0]..name}")
if [ -z $SA_SECRET_NAME ]
then
# Create the secret and bind it to the desired SA
kubectl apply -f - <<EOF
apiVersion: v1
kind: "**********"
type: "**********"
metadata:
  name: ${TELEPORT_SA}
  namespace: ${NAMESPACE}
  annotations:
    kubernetes.io/service-account.name: "${TELEPORT_SA}"
EOF

SA_SECRET_NAME= "**********"
fi

# Note: "**********"
# be plaintext in kubeconfig.
SA_TOKEN= "**********"={.data['token']}" | base64 ${BASE64_DECODE_FLAG})
CA_CERT= "**********"={.data['ca\.crt']}")

# Extract cluster IP from the current context
CURRENT_CONTEXT=$(kubectl config current-context)
CURRENT_CLUSTER=$(kubectl config view -o jsonpath="{.contexts[?(@.name == \"${CURRENT_CONTEXT}\"})].context.cluster}")
CURRENT_CLUSTER_ADDR=$(kubectl config view -o jsonpath="{.clusters[?(@.name == \"${CURRENT_CLUSTER}\"})].cluster.server}")

echo "Writing kubeconfig."
cat > kubeconfig <<EOF
apiVersion: v1
clusters:
- cluster:
    certificate-authority-data: ${CA_CERT}
    server: ${CURRENT_CLUSTER_ADDR}
  name: ${CURRENT_CLUSTER}
contexts:
- context:
    cluster: ${CURRENT_CLUSTER}
    user: ${CURRENT_CLUSTER}-${TELEPORT_SA}
  name: ${CURRENT_CONTEXT}
current-context: ${CURRENT_CONTEXT}
kind: Config
preferences: {}
users:
- name: ${CURRENT_CLUSTER}-${TELEPORT_SA}
  user:
    token: "**********"
EOF

echo "---
Done!

Copy the generated kubeconfig file to your server, and set the
KUBECONFIG env variable or move the file to /.kube/config.

Note: Kubernetes RBAC rules were created, you won't need to create them manually."EN}
EOF

echo "---
Done!

Copy the generated kubeconfig file to your server, and set the
KUBECONFIG env variable or move the file to /.kube/config.

Note: Kubernetes RBAC rules were created, you won't need to create them manually."