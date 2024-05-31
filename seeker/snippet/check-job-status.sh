#date: 2024-05-31T17:10:50Z
#url: https://api.github.com/gists/b099e8df77ac9d83eec5bad21b27f1cf
#owner: https://api.github.com/users/nyrahul

kubectl get workload -A  -o json | jq '.items[] | "\(.metadata.namespace) \(.metadata.name) \(.status.conditions[-1].type)"'