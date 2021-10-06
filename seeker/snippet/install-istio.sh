#date: 2021-10-06T17:13:43Z
#url: https://api.github.com/gists/6b51fbf695ec9c95829bf0fe4b833bf6
#owner: https://api.github.com/users/rdpetrusek

#!/bin/bash

# https://istio.io/latest/docs/setup/install/helm/

kubectl create namespace istio-system

helm install istio-base manifests/charts/base -n istio-system
sleep 5

helm install istiod manifests/charts/istio-control/istio-discovery -n istio-system
sleep 5

helm install istio-ingress manifests/charts/gateways/istio-ingress -n istio-system