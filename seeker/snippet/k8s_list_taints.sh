#date: 2024-05-06T17:03:12Z
#url: https://api.github.com/gists/ce37a06ae38733cc976f42756c8fb1fe
#owner: https://api.github.com/users/rjchicago

# list taints (all nodes)
kubectl get nodes -o json | jq -r '.items[] | .metadata.name + " " + .spec.taints[]?.key + "=" + .spec.taints[]?.key + ":" + .spec.taints[]?.effect'

# list taints (exclude control plane)
kubectl get nodes -o json -l '!node-role.kubernetes.io/control-plane' | jq -r '.items[] | .metadata.name + " " + .spec.taints[]?.key + "=" + .spec.taints[]?.key + ":" + .spec.taints[]?.effect'

# remove taint example (note: the hyphen after the taint to remove)
# kubectl taint nodes $NODE $TAINT-