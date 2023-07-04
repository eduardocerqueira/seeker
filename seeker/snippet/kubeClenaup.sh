#date: 2023-07-04T17:09:01Z
#url: https://api.github.com/gists/f5b96789e3f65e6bd29c2e73baa0c0eb
#owner: https://api.github.com/users/rammanokar

#!/bin/bash

# Function to get all namespaces
get_namespaces() {
    kubectl get namespaces --no-headers -o custom-columns=NAME:.metadata.name | grep -Ev "^(kube-system|kube-public|kube-node-lease|istio.*|gke.*)$"
}

# Function to get pods based on condition
get_pods() {
    local namespace=$1
    local condition=$2
    kubectl get pods -n $namespace -o json | jq -r ".items[] | select($condition) | .metadata.name"
}

# Function to scale down deployment
scale_down() {
    local namespace=$1
    local pod=$2
    local deployment_name=${pod%%-*}
    kubectl scale deployment -n $namespace "$deployment_name-service" --replicas=0 && echo "Scaling down $deployment_name in $namespace"
}

# Main script
main() {
    local namespaces=$(get_namespaces)

    for namespace in $namespaces; do
        echo "Processing namespace: $namespace"

        local not_ready_pods=$(get_pods $namespace 'any(.status.containerStatuses[]; .ready==false)')
        for pod in $not_ready_pods; do
            if [[ $pod != *"cron"* ]]; then
                scale_down $namespace $pod
            fi
        done

        local crashloopbackoff_pods=$(get_pods $namespace '.status.phase=="CrashLoopBackOff"')
        for pod in $crashloopbackoff_pods; do
            if [[ $pod != *"cron"* ]]; then
                scale_down $namespace $pod
            fi
        done
    done
}

main "$@"