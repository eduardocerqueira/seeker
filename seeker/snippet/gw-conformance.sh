#date: 2025-08-20T17:00:52Z
#url: https://api.github.com/gists/a56512ad5bf4a4fe67bba024dbc1d279
#owner: https://api.github.com/users/rikatz

#!/usr/bin/env bash
#
kind create cluster

kubectl apply -f https://raw.githubusercontent.com/metallb/metallb/v0.15.2/config/manifests/metallb-native.yaml
kubectl wait --timeout=5m deploy -n metallb-system controller --for=condition=Available

kubectl apply -f - <<EOF
apiVersion: metallb.io/v1beta1
kind: IPAddressPool
metadata:
  namespace: metallb-system
  name: kube-services
spec:
  addresses:
  - 172.18.200.100-172.18.200.150
---
apiVersion: metallb.io/v1beta1
kind: L2Advertisement
metadata:
  name: kube-services
  namespace: metallb-system
spec:
  ipAddressPools:
  - kube-services
EOF


kubectl apply --server-side -f https://github.com/envoyproxy/gateway/releases/download/v1.5.0/install.yaml
kubectl wait --timeout=5m -n envoy-gateway-system deployment/envoy-gateway --for=condition=Available
kubectl apply -f https://github.com/envoyproxy/gateway/releases/download/v1.5.0/quickstart.yaml -n default