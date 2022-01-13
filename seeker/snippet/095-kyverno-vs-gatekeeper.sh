#date: 2022-01-13T16:56:25Z
#url: https://api.github.com/gists/d2b7f4fbf5c739eb8468180c085e6c20
#owner: https://api.github.com/users/andrzej-natzka

###############################################
# OPA with Gatekeeper vs. Kyverno             #
# Kubernetes Policy Management Tools Compared #
# https://youtu.be/9gSrRNmmKBc                #
###############################################

# Referenced videos:
# - Kubernetes-Native Policy Management With Kyverno: https://youtu.be/DREjzfTzNpA
# - How to apply policies in Kubernetes using Open Policy Agent (OPA) and Gatekeeper: https://youtu.be/14lGc7xMAe4
# - K3d - How to run Kubernetes cluster locally using Rancher k3s: https://youtu.be/mCesuGk-Fks
# - Kustomize - How to Simplify Kubernetes Configuration Management: https://youtu.be/Twtbg6LFnAg

#########
# Setup #
#########

git clone https://github.com/vfarcic/gatekeeper-vs-kyverno-demo

cd gatekeeper-vs-kyverno-demo

# Please watch https://youtu.be/mCesuGk-Fks if you are not familiar with k3d.
# It could be any other Kubernetes cluster. It does not have to be k3d.
k3d cluster create --config k3d.yaml

kubectl create namespace prod-gatekeeper

kubectl create namespace prod-kyverno

#################
# Setup Kyverno #
#################

kubectl create namespace kyverno

kubectl apply \
    --filename https://raw.githubusercontent.com/kyverno/kyverno/main/definitions/release/install.yaml

####################
# Setup Gatekeeper #
####################

kubectl apply \
    --filename https://raw.githubusercontent.com/open-policy-agent/gatekeeper/release-3.3/deploy/gatekeeper.yaml

##########################
# Libraries and examples #
##########################

# Gatekeeper
# Open https://open-policy-agent.github.io/gatekeeper/website/docs/library

# Gatekeeper
# Open https://github.com/open-policy-agent/gatekeeper-library/blob/master/library/general/block-nodeport-services/samples/block-node-port/constraint.yaml

# Kyverno
# Open https://kyverno.io/policies

# Kyverno
# Open https://kyverno.io/policies/best-practices/restrict_node_port/restrict_node_port/

# Gatekeeper
# You might want to watch https://youtu.be/Twtbg6LFnAg if you are not familiar with Kustomize
kustomize build \
    github.com/open-policy-agent/gatekeeper-library/library \
    | kubectl apply --filename -

# Kyverno
# No templates needed

# Gatekeeper
kubectl apply --filename gatekeeper

# Kyverno
kubectl apply --filename kyverno

####################
# Writing policies #
####################

# Gatekeeper
# Open https://github.com/open-policy-agent/gatekeeper-library/blob/master/library/general/block-nodeport-services/template.yaml

# Gatekeeper
# Open https://github.com/open-policy-agent/gatekeeper-library/blob/master/library/general/block-nodeport-services/samples/block-node-port/constraint.yaml

# Kyverno
# Open https://kyverno.io/policies/best-practices/restrict_node_port/restrict_node_port/

# Gatekeeper
# Open https://github.com/open-policy-agent/gatekeeper-library/blob/master/library/general/containerresourceratios/template.yaml

# Gatekeeper
# Open https://github.com/open-policy-agent/gatekeeper-library/blob/master/library/general/containerresourceratios/samples/container-must-meet-ratio/constraint.yaml

# Kyverno
# Open https://kyverno.io/policies/best-practices/require_pod_requests_limits/require_pod_requests_limits/

######################
# Enforcing policies #
######################

cat app.yaml

# Gatekeeper
kubectl --namespace prod-gatekeeper \
    apply --filename app.yaml

# Gatekeeper
kubectl --namespace prod-gatekeeper \
    get deployments

# Gatekeeper
kubectl --namespace prod-gatekeeper \
    describe replicaset \
    --selector app=devops-toolkit

# Kyverno
kubectl --namespace prod-kyverno \
    apply --filename app.yaml

#############
# Reporting #
#############

# Kyverno
kubectl get policyreports -A

###########
# Destroy #
###########

k3d cluster delete devops-toolkit
