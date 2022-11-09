#date: 2022-11-09T17:09:21Z
#url: https://api.github.com/gists/33c399c57ef29eac3c64a44c0e3cab28
#owner: https://api.github.com/users/alioualarbi

#!/usr/bin/env bash

#####################################################################
# REFERENCES
# - https://cloud.google.com/kubernetes-engine/docs/add-on/backup-for-gke/concepts/backup-for-gke
# - https://cloud.google.com/kubernetes-engine/docs/add-on/backup-for-gke/how-to/install 
#####################################################################

export PROJECT_ID=$(gcloud config get-value project)
export PROJECT_USER=$(gcloud config get-value core/account) # set current user
export PROJECT_NUMBER=$(gcloud projects describe $PROJECT_ID --format="value(projectNumber)")
export IDNS=${PROJECT_ID}.svc.id.goog # workload identity domain

export GCP_REGION="us-central1" # CHANGEME (OPT)
export GCP_ZONE="us-central1-a" # CHANGEME (OPT)

export NETWORK_NAME="default"

# enable apis
gcloud services enable compute.googleapis.com \
    storage.googleapis.com \
    container.googleapis.com \
    gkebackup.googleapis.com

# configure gcloud sdk
gcloud config set compute/region $GCP_REGION
gcloud config set compute/zone $GCP_ZONE


#######################################################################
# CLUSTER
#######################################################################
export CLUSTER_NAME="central"

# create regional cluster with 3 nodes (1 per zone)
gcloud beta container clusters create $CLUSTER_NAME \
    --project=$PROJECT_ID  \
    --region=$GCP_REGION \
    --addons=BackupRestore \
    --num-nodes=1 \
    --enable-autoupgrade --no-enable-basic-auth \
    --no-issue-client-certificate --enable-ip-alias \
    --metadata disable-legacy-endpoints=true \
    --workload-pool=$IDNS


#######################################################################
# SAMPLE APP
# - https://kubernetes.io/docs/tutorials/stateful-application/mysql-wordpress-persistent-volume/ 
#######################################################################
# create Kustomize file
cat > ./kustomization.yaml << EOF
 "**********"s "**********"e "**********"c "**********"r "**********"e "**********"t "**********"G "**********"e "**********"n "**********"e "**********"r "**********"a "**********"t "**********"o "**********"r "**********": "**********"
- name: mysql-pass
  literals:
  - password= "**********"
EOF

# download manifests
curl -LO https://k8s.io/examples/application/wordpress/mysql-deployment.yaml
curl -LO https://k8s.io/examples/application/wordpress/wordpress-deployment.yaml 

# update Kustomize file (note >> which appends original file)
cat >> ./kustomization.yaml << EOF
resources:
  - mysql-deployment.yaml
  - wordpress-deployment.yaml
EOF

# deploy (using built-in kustomize feature of kubectl)
kubectl apply -k ./

# get service external IP address
export WORDPRESS_IP=$(kubectl get svc wordpress -o jsonpath='{.status.loadBalancer.ingress[0].ip}')

# visit page and set up site (to avoid getting exploited)


#######################################################################
# BACKUP
#######################################################################
export BACKUP_PLAN="wordpress-backup"
export SCHEDULE="0 * * * *"  # every hour at minute 0
export LOCATION="us-west1"   # someplace different than current region
export CLUSTER="projects/$PROJECT_ID/locations/$GCP_REGION/clusters/$CLUSTER_NAME"
export RETAIN_DAYS="7"
# omitted ENCRYPTION_KEY, DELETE_LOCK_DAYS

# view backup locations
gcloud alpha container backup-restore locations list \
    --project $PROJECT_ID

# create backup plan
gcloud alpha container backup-restore backup-plans create $BACKUP_PLAN \
    --project=$PROJECT_ID \
    --location=$LOCATION \
    --cluster=$CLUSTER \
    --all-namespaces \
    --include-secrets \
    --include-volume-data \
    --cron-schedule=$SCHEDULE \
    --backup-retain-days=$RETAIN_DAYS \
    --locked

# verify
gcloud alpha container backup-restore backup-plans list \
    --project=$PROJECT_ID \
    --location=$LOCATION

# back up workload manually
export BACKUP="manual-backup1"

gcloud alpha container backup-restore backups create $BACKUP \
    --project=$PROJECT_ID \
    --location=$LOCATION \
    --backup-plan=$BACKUP_PLAN \
    --wait-for-completion


#######################################################################
# RESTORE
# - https://cloud.google.com/sdk/gcloud/reference/alpha/container/backup-restore/restore-plans/create 
#######################################################################
export RESTORE_LOCATION=$GCP_REGION # same as cluster where PVCs exist
export RESTORE_PLAN="wordpress-restore"
export CLUSTER_RESOURCE_CONFLICT_POLICY="use-backup-version" # want to overwrite with backup to prove it works
export NAMESPACED_RESOURCE_RESTORE_MODE="delete-and-restore" # force recreate to prove it works

# delete application from cluster
kubectl delete svc/wordpress svc/wordpress-mysql deployment/wordpress-mysql deployment/wordpress

# create restore plan
gcloud alpha container backup-restore restore-plans create $RESTORE_PLAN \
    --project=$PROJECT_ID \
    --location=$RESTORE_LOCATION \
    --backup-plan=projects/$PROJECT_ID/locations/$LOCATION/backupPlans/$BACKUP_PLAN \
    --cluster=$CLUSTER \
    --namespaced-resource-restore-mode=$NAMESPACED_RESOURCE_RESTORE_MODE \
    --all-namespaces

# verify
gcloud alpha container backup-restore restore-plans list \
    --project=$PROJECT_ID \
    --location=$LOCATION

# perform restore to new cluster
export RESTORE="manual-restore1"

gcloud alpha container backup-restore restores create $RESTORE \
    --project=$PROJECT_ID \
    --location=$GCP_REGION \
    --restore-plan=$RESTORE_PLAN \
    --backup=projects/$PROJECT_ID/locations/$LOCATION/backupPlans/$BACKUP_PLAN/backups/$BACKUP

# NOTE: will not work with different cluster because PVC
# get restored wordpress external IP


# test in browser and verify custom post exists
