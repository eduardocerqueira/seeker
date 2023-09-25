#date: 2023-09-25T16:44:46Z
#url: https://api.github.com/gists/346834e64e43aead01605e14cf2df4e8
#owner: https://api.github.com/users/tuxerrante

#!/bin/bash

#
# download yq
#

curl -fsSL https://github.com/mikefarah/yq/releases/download/v4.9.1/yq_linux_amd64 -o /usr/local/bin/yq
chmod +x /usr/local/bin/yq

#
# get certs from kubeconfig
#

KEY_DATA="$(yq eval '.users[] | select(.name == "kubernetes-admin") | .user.client-key-data' ~/.kube/config | base64 -d)"
CERT_DATA="$(yq eval '.users[] | select(.name == "kubernetes-admin") | .user.client-certificate-data' ~/.kube/config | base64 -d)"
CA_DATA="$(yq eval '.clusters[] | select(.name == "kubernetes") | .cluster.certificate-authority-data' ~/.kube/config | base64 -d)"

#
# deploy pod
#

curl -X POST \
  --cacert <(echo "${CA_DATA}") \
  --cert <(echo "${CERT_DATA}") \
  --key <(echo "${KEY_DATA}") \
  -H "Content-Type: application/json" \
  -H "Accept: application/json" \
  --data-binary @<(yq eval -j pod.yaml) \
  https://192.168.50.101:6443/api/v1/namespaces/default/pods?fieldManager=kubectl-create

#
# list pods
#

curl \
  --cacert <(echo "${CA_DATA}") \
  --cert <(echo "${CERT_DATA}") \
  --key <(echo "${KEY_DATA}") \
  -H "Accept: application/json" \
  https://192.168.50.101:6443/api/v1/namespaces/default/pods?limit=5

#
# delete pod
#

curl -X DELETE \
  --cacert <(echo "${CA_DATA}") \
  --cert <(echo "${CERT_DATA}") \
  --key <(echo "${KEY_DATA}") \
  -H "Accept: application/json" \
  -H "Content-Type: application/json" \
  https://192.168.50.101:6443/api/v1/namespaces/default/pods/pod
