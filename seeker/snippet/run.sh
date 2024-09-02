#date: 2024-09-02T16:54:10Z
#url: https://api.github.com/gists/a9dc47e7924feb6dc3ad32ac13751ffb
#owner: https://api.github.com/users/hrivera-ntap

#!/usr/bin/env sh

set -x

kubectl get pods -A --selector app=tekton-pipelines-controller,pipeline.tekton.dev/release --show-labels
echo ""
sleep 5

kubectl delete -f tekton-pipeline.yaml || true
sleep 10
kubectl apply -f tekton-pipeline.yaml
sleep 30

# debug
kubectl describe pipelineruns -n tekton-timeout-bug
echo ""
kubectl get taskruns -n tekton-timeout-bug
echo ""
kubectl get pipelineruns -n tekton-timeout-bug
echo ""