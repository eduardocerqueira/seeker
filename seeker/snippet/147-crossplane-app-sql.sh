#date: 2022-01-28T16:53:56Z
#url: https://api.github.com/gists/2be27383955a4a2914cb782d887f3097
#owner: https://api.github.com/users/vfarcic

# Source: https://gist.github.com/2be27383955a4a2914cb782d887f3097

####################################################################################
# How To Manage Applications With Databases Using Helm, Crossplane, And Schemahero #
# https://youtu.be/I4TfHqONeKg                                                     #
####################################################################################

# Additional Info:
# - Crossplane - GitOps-based Infrastructure as Code through Kubernetes API: https://youtu.be/n8KjVmuHm7A
# - How To Shift Left Infrastructure Management Using Crossplane Compositions: https://youtu.be/AtbS1u2j7po
# - How To Manage Kubernetes Applications Using Crossplane Compositions: https://youtu.be/eIQpGXUGEow
# - Terraform vs. Pulumi vs. Crossplane - Infrastructure as Code (IaC) Tools Comparison: https://youtu.be/RaoKcJGchKM
# - SchemaHero - Database Schema Migrations Inside Kubernetes: https://youtu.be/SofQxb4CDQQ
# - Bitnami Sealed Secrets - How To Store Kubernetes Secrets In Git Repositories: https://youtu.be/xd2QoV6GJlc

#########
# Setup #
#########

# Using Rancher Desktop for the demo, but it can be any other Kubernetes cluster with Ingress

# If not using Rancher Desktop, replace `127.0.0.1` with the base host accessible through Ingress
export INGRESS_HOST=127.0.0.1

git clone \
    https://github.com/vfarcic/devops-toolkit-crossplane

cd devops-toolkit-crossplane

kubectl create namespace a-team

kubectl create namespace crossplane-system

cat charts/sql/values.yaml \
    | sed -e "s@host: .*@host: devops-toolkit.$INGRESS_HOST.nip.io@g" \
    | tee charts/sql/values.yaml

cat examples/app-backend-sql.yaml \
    | sed -e "s@host: .*@host: devops-toolkit.$INGRESS_HOST.nip.io@g" \
    | tee examples/app-backend-sql.yaml

#############
# Setup GCP #
#############

export PROJECT_ID=devops-toolkit-$(date +%Y%m%d%H%M%S)

gcloud projects create $PROJECT_ID

echo "https://console.cloud.google.com/billing/enable?project=$PROJECT_ID"

# Set the billing account

echo "https://console.cloud.google.com/apis/library/sqladmin.googleapis.com?project=$PROJECT_ID"

# Open the URL and *ENABLE API*

export SA_NAME=devops-toolkit

export SA="${SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"

gcloud iam service-accounts \
    create $SA_NAME \
    --project $PROJECT_ID

export ROLE=roles/admin

gcloud projects add-iam-policy-binding \
    --role $ROLE $PROJECT_ID \
    --member serviceAccount:$SA

gcloud iam service-accounts keys \
    create gcp-creds.json \
    --project $PROJECT_ID \
    --iam-account $SA

kubectl --namespace crossplane-system \
    create secret generic gcp-creds \
    --from-file creds=./gcp-creds.json

cat crossplane-config/provider-config-gcp.yaml \
    | sed -e "s@projectID: .*@projectID: $PROJECT_ID@g" \
    | tee crossplane-config/provider-config-gcp.yaml

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
    --filename crossplane-config/provider-sql.yaml

kubectl apply \
    --filename crossplane-config/config-sql.yaml

kubectl apply \
    --filename crossplane-config/config-app.yaml

kubectl apply \
    --filename crossplane-config/provider-kubernetes-incluster.yaml

kubectl apply \
    --filename crossplane-config/provider-config-gcp.yaml

# Please re-run the previous command if the output is `unable to recognize ...`

####################
# Setup SchemaHero #
####################

kubectl krew install schemahero

kubectl schemahero install

########
# Demo #
########

helm upgrade --install \
    sql-demo charts/sql/. \
    --namespace a-team \
    --create-namespace

kubectl --namespace a-team \
    get pods

kubectl --namespace a-team \
    describe pod --selector app=sql-demo

cat charts/sql/templates/app.yaml

kubectl get managed

kubectl --namespace a-team \
    get pods

curl "http://devops-toolkit.$INGRESS_HOST.nip.io/addVideo?id=RaoKcJGchKM&name=Terraform+vs+Pulumi+vs+Crossplane&url=https://youtu.be/RaoKcJGchKM"

kubectl --namespace a-team logs \
    --selector app=sql-demo

kubectl --namespace a-team \
    get secrets

export DB_ENDPOINT=$(kubectl \
    --namespace a-team \
    get secret sql-demo \
    --output jsonpath="{.data.endpoint}" \
    | base64 -d)

export DB_PASS=$(kubectl \
    --namespace a-team \
    get secret sql-demo \
    --output jsonpath="{.data.password}" \
    | base64 -d)

helm upgrade --install \
    sql-demo charts/sql/. \
    --namespace a-team \
    --set schema.endpoint=$DB_ENDPOINT \
    --set schema.password=$DB_PASS \
    --wait

curl "http://devops-toolkit.$INGRESS_HOST.nip.io/addVideo?id=RaoKcJGchKM&name=Terraform+vs+Pulumi+vs+Crossplane&url=https://youtu.be/RaoKcJGchKM"

curl "http://devops-toolkit.$INGRESS_HOST.nip.io/addVideo?id=yrj4lmScKHQ&name=Crossplane+with+Terraform&url=https://youtu.be/yrj4lmScKHQ"

curl "http://devops-toolkit.$INGRESS_HOST.nip.io/getVideos"

cat charts/sql/templates/db.yaml

cat charts/sql/templates/schema.yaml

cat charts/sql/templates/app.yaml

cat charts/sql/templates/crossplane-app.yaml

helm upgrade --install \
    sql-demo charts/sql/. \
    --namespace a-team \
    --set crossplaneApp=true \
    --reuse-values \
    --wait

kubectl --namespace a-team get appclaims

kubectl --namespace a-team get all,ingresses

curl "http://devops-toolkit.$INGRESS_HOST.nip.io/getVideos"

helm delete sql-demo --namespace a-team

cat examples/app-backend-sql.yaml

kubectl --namespace a-team apply \
    --filename examples/app-backend-sql.yaml

kubectl --namespace a-team \
    get all,ingresses,appclaims,cloudsqlinstances

cat packages/sql/definition.yaml

cat packages/sql/google.yaml

###########
# Destroy #
###########

kubectl --namespace a-team delete \
    --filename examples/app-backend-sql.yaml

kubectl get cloudsqlinstances

# Repeat the previous command until all the resources are deleted

gcloud projects delete $PROJECT_ID

# Destroy or reset the management cluster
