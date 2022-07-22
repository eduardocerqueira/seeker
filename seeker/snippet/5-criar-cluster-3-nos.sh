#date: 2022-07-22T17:11:24Z
#url: https://api.github.com/gists/0fe9c7c7e0a35d81fa71dd0a46774eea
#owner: https://api.github.com/users/AlexGalhardo

kind create cluster --name demo-cluster-3 --config ./4-kind-3-nodes.yml
kind get clusters
kubectl config get-contexts
kubectl get pods -n kube-system
kubectl get nodes