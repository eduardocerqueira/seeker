#date: 2024-04-19T17:07:57Z
#url: https://api.github.com/gists/caf14a5fddef540edc33b79fbc8db40b
#owner: https://api.github.com/users/qudongfang

for c in $(kubectx | grep dev); do
    echo $c;
    kubectx $c

    kubectl delete deployments istiod -n istio-system
    kubectl delete hpa istiod -n istio-system
    kubectl delete pdb istiod -n istio-system
    kubectl delete svc istiod -n istio-system  --wait=false
    kubectl delete cm -n istio-system istio
done