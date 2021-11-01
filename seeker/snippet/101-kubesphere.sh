#date: 2021-11-01T16:57:39Z
#url: https://api.github.com/gists/0e362b324323632df3a9927c2b7f1037
#owner: https://api.github.com/users/ItsKDaniel

# Source: https://gist.github.com/511be70015196e9e7f0ce2a81f5b36e5

###############################################################
# Kubesphere                                                  #
# Kubernetes Platform For Cloud-Native Application Management #
# https://youtu.be/1OOLeCVWTXE                                #
###############################################################

# Referenced videos:
# - Lens - Kubernetes IDE and Dashboard: https://youtu.be/q_ooC1xcGCg
# - Running Jenkins In Kubernetes - Tutorial And Review: https://youtu.be/2Kc3fUJANAc
# - Bitnami Kubeapps - Application Dashboard for Kubernetes: https://youtu.be/DJ_k5fhODi0

# Feel free to use any other Kubernetes platform
minikube start --memory 6g --cpus 4

# If not using Minikube, install Ingress in whichever way is appropriate for the Kubernetes distribution
minikube addons enable ingress

# If not using Minikube, install Metrics Server in whichever way is appropriate for the Kubernetes distribution, unless it is already installed.
minikube addons enable metrics-server

kubectl apply \
    --filename https://github.com/kubesphere/ks-installer/releases/download/v3.2.0/kubesphere-installer.yaml

kubectl apply \
    --filename https://github.com/kubesphere/ks-installer/releases/download/v3.2.0/cluster-configuration.yaml

kubectl --namespace kubesphere-system \
    logs $(kubectl \
    --namespace kubesphere-system \
    get pod \
    --selector app=ks-install \
    --output jsonpath='{.items[0].metadata.name}') \
    --follow

# Copy the `Console` address from the output and paste it instead of `[...]` in the command that follows.
export CONSOLE_ADDR=[...]

kubectl get pods --all-namespaces

kubectl top pods --all-namespaces

echo $CONSOLE_ADDR

# Use `admin` as the Username and `P@88w0rd` as the Password

kubectl get namespaces

kubectl --namespace my-project get all

kubectl get svc --all-namespaces

kubectl --namespace kubesphere-controls-system \
    describe pod \
    --selector component=kubesphere-router

# Use `vfarcic/devops-toolkit-series` as the image

kubectl --namespace my-project \
    get all,ingresses

# Set `spec.devops.enabled` to `true`

kubectl --namespace kubesphere-system \
    logs $(kubectl \
    --namespace kubesphere-system \
    get pod \
    --selector app=ks-install \
    --output jsonpath='{.items[0].metadata.name}') \
    --follow

kubectl top pods --all-namespaces

# Set `spec.openpitrix.store.enabled` to `true`

kubectl --namespace kubesphere-system \
    logs $(kubectl \
    --namespace kubesphere-system \
    get pod \
    --selector app=ks-install \
    --output jsonpath='{.items[0].metadata.name}') \
    --follow

minikube delete
