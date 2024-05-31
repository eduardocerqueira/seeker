#date: 2024-05-31T17:01:28Z
#url: https://api.github.com/gists/b92a80c2251c7a54d1feada320c5eac6
#owner: https://api.github.com/users/jm96441n

#!/bin/bash

set -e

export CONSUL_K8S_CHARTS_LOCATION="$HOME/hashi/consul-k8s/charts/consul"

if [ -z "$(kind get clusters | rg "basic")" ]; then
	kind create cluster --config cluster.yaml
fi

# The following line assumes that you have compiled the image locally using `make docker/dev` from the consul-k8s repo
kind load docker-image consul-k8s-control-plane:local -n basic
kind load docker-image consul:local -n basic

kubectl create namespace consul
echo "helm installing"
helm upgrade --install consul $CONSUL_K8S_CHARTS_LOCATION -f ./consul_values.yaml -n consul --create-namespace --wait
echo "helm is done"
kubectl wait --timeout=180s --for=condition=Available=True deployments/consul-consul-connect-injector -n consul
kubectl apply -f ./proxy-defaults.yaml
kubectl apply -f ./gw.yaml
kubectl apply -f ./gw2.yaml
# while ! kubectl get deployments api-gateway; do sleep 1; done
kubectl wait --timeout=180s --for=condition=Available=True deployments/api-gateway || true
kubectl get svc -n consul
