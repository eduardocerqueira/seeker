#date: 2021-11-30T17:01:51Z
#url: https://api.github.com/gists/c760efbdb28ce9cc2481a8115610cc49
#owner: https://api.github.com/users/vfarcic

#############################################################################
# How To Package And Distribute Crossplane Compositions As Container Images #
# https://youtu.be/i7MFiInJV8c                                              #
#############################################################################

# Referenced videos:
# - How To Shift Left Infrastructure Management Using Crossplane Composites: https://youtu.be/AtbS1u2j7po

#################
# Setup Cluster #
#################

# Watch https://youtu.be/BII6ZY2Rnlc if you are not familiar with GitHub CLI
gh repo fork vfarcic/devops-toolkit-crossplane \
    --clone

cd devops-toolkit-crossplane

# Using Rancher Desktop for the demo, but it can be any other Kubernetes cluster with Ingress

# If not using Rancher Desktop, replace `127.0.0.1` with the base host accessible through NGINX Ingress
export INGRESS_HOST=127.0.0.1

kubectl create namespace crossplane-system

kubectl create namespace a-team

# Replace `[...]` with your Docker Hub user
export DH_USER=[...]

#############
# Setup AWS #
#############

# Replace `[...]` with your access key ID`
export AWS_ACCESS_KEY_ID=[...]

# Replace `[...]` with your secret access key
export AWS_SECRET_ACCESS_KEY=[...]

echo "[default]
aws_access_key_id = $AWS_ACCESS_KEY_ID
aws_secret_access_key = $AWS_SECRET_ACCESS_KEY
" >aws-creds.conf

kubectl --namespace crossplane-system \
    create secret generic aws-creds \
    --from-file creds=./aws-creds.conf

####################
# Setup Crossplane #
####################

helm repo add crossplane-stable \
    https://charts.crossplane.io/stable

helm repo update

helm upgrade --install \
    crossplane crossplane-stable/crossplane \
    --namespace crossplane-system \
    --create-namespace \
    --wait

kubectl apply \
    --filename crossplane-config/provider-aws.yaml

kubectl apply \
    --filename crossplane-config/provider-config-aws.yaml

# Please re-run the previous command if the output is `unable to recognize ...`

kubectl apply \
    --filename crossplane-config/provider-helm.yaml

kubectl apply \
    --filename crossplane-config/provider-kubernetes.yaml

##############################################
# Creating Crossplane Configuration Packages #
##############################################

cd packages/k8s

cat definition.yaml

cat eks.yaml

cat aks.yaml

cat gke.yaml

cat civo.yaml

cat crossplane.yaml

curl -sL https://raw.githubusercontent.com/crossplane/crossplane/release-1.5/install.sh \
    | sh

kubectl crossplane build configuration \
    --name k8s

# Requires `~/.docker/config.json` file (no need for Docker binary)
kubectl crossplane push configuration \
    $DH_USER/crossplane-k8s:v0.1.4

cd ../../

cat crossplane-config/config-k8s.yaml

kubectl apply \
    --filename crossplane-config/config-k8s.yaml

kubectl get pkgrev

cat examples/aws-eks.yaml

kubectl --namespace a-team apply \
    --filename examples/aws-eks.yaml

kubectl get managed

###########
# Destroy #
###########

kubectl --namespace a-team delete \
    --filename examples/aws-eks.yaml

kubectl get managed

# Repeat the previous command until all the managed resources are removed

# Destroy or reset the management cluster

# Delete the GitOps repo
