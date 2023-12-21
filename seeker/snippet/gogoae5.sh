#date: 2023-12-21T16:54:49Z
#url: https://api.github.com/gists/e8a422e2070daab85eceee6ad9ba398c
#owner: https://api.github.com/users/brian-anaconda

#!/bin/bash

PROFILE=sandbox
SSH_KEY=bbartels
REGION=us-east-1

HELM_CHART_DIR=ae-helm-chart-5.6.2

CLUSTER_NAME=brian-ae5-authnfs-test26
AE5_HOSTNAME=${CLUSTER_NAME}.sb.anacondaconnect.com

SUBNET_IDS='subnet-097282935fc846ce3,subnet-0a7de4492780a4180'
SEC_GROUP_IDS=sg-0baaeaa5393d2b16a

AWS_ACCOUNT_ID=621741996708
CONTAINER_IMAGE_REGISTRY=602401143452.dkr.ecr.us-east-1.amazonaws.com
ROLE_ARN=arn:aws:iam::621741996708:role/tmp_integration_eng_eks_admin_role

K8S_VERSION=1.26
AMI_TYPE=AL2_x86_64
INSTANCE_TYPE=m5.2xlarge
NUM_NODES=2

proro () { echo -n "--profile $PROFILE --region $REGION "; }
prettyo () { echo -n " --no-cli-pager --output table "; }
ppo () { proro; prettyo; }
jo () { echo -n " --output json "; }
pj () { proro; jo; }

spin='-\|/'
spintil () {
  echo -n "   Waiting for ${3}..."
  until [ "$(eval "$1")" == "$2" ]; do
    i=$(( (i+1) %4 ))
    printf "\r${spin:$i:1}"
    sleep 2
  done
  echo; echo "Done"
}

aws eks create-cluster $(ppo) \
  --name $CLUSTER_NAME \
  --kubernetes-version $K8S_VERSION \
  --role-arn $ROLE_ARN \
  --resources-vpc-config subnetIds=$SUBNET_IDS,securityGroupIds=$SEC_GROUP_IDS

cluster_status="aws eks describe-cluster $(proro) --name $CLUSTER_NAME|jq -r .cluster.status"
spintil "$cluster_status" "ACTIVE" "cluster"

aws eks create-nodegroup $(ppo) \
  --cluster-name $CLUSTER_NAME \
  --nodegroup-name ${CLUSTER_NAME}_nodegroup \
  --subnets $(echo $SUBNET_IDS|cut -d, -f1) \
  --node-role $ROLE_ARN \
  --ami-type $AMI_TYPE \
  --instance-types $INSTANCE_TYPE \
  --disk-size 100 \
  --remote-access ec2SshKey=$SSH_KEY \
  --scaling-config minSize=1,maxSize=${NUM_NODES},desiredSize=${NUM_NODES}

nodegroup_status="aws eks describe-nodegroup --cluster-name $CLUSTER_NAME --nodegroup-name ${CLUSTER_NAME}_nodegroup $(pj)|jq -r .nodegroup.status"
spintil "$nodegroup_status" "ACTIVE" "node group"

aws eks update-kubeconfig --name $CLUSTER_NAME $(proro)
kubectl config use-context arn:aws:eks:${REGION}:${AWS_ACCOUNT_ID}:cluster/${CLUSTER_NAME}

trues="$(for (( i=1; i<=$NUM_NODES; i++ )); do echo True;done|xargs)"
nodes_status="kubectl get nodes -o custom-columns='Ready:.status.conditions[3].status' --no-headers|xargs"
spintil "$nodes_status" "$trues" "2 nodes"

kubectl get nodes

kubectl create ns anaconda-enterprise
kubectl config set-context --current --namespace=anaconda-enterprise

cat <<EOT > iam-policy.json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "elasticfilesystem:DescribeAccessPoints",
        "elasticfilesystem:DescribeFileSystems"
      ],
      "Resource": "*"
    },
    {
      "Effect": "Allow",
      "Action": [
        "elasticfilesystem:CreateAccessPoint"
      ],
      "Resource": "*",
      "Condition": {
        "StringLike": {
          "aws:RequestTag/efs.csi.aws.com/cluster": "true"
        }
      }
    },
    {
      "Effect": "Allow",
      "Action": "elasticfilesystem:DeleteAccessPoint",
      "Resource": "*",
      "Condition": {
        "StringEquals": {
          "aws:ResourceTag/efs.csi.aws.com/cluster": "true"
        }
      }
    }
  ]
}
EOT

aws iam create-policy $(ppo) --policy-name ${CLUSTER_NAME}_AmazonEKS_EFS_CSI_Driver_Policy --policy-document file://iam-policy.json

CLUSTER_OIDC=$(aws eks describe-cluster $(proro) --name $CLUSTER_NAME --query "cluster.identity.oidc.issuer" --output text|rev|cut -d/ -f1|rev)

cat <<EOT > trust-policy.json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Federated": "arn:aws:iam::${AWS_ACCOUNT_ID}:oidc-provider/oidc.eks.us-east-1.amazonaws.com/id/${CLUSTER_OIDC}"
      },
      "Action": "sts:AssumeRoleWithWebIdentity",
      "Condition": {
        "StringEquals": {
          "oidc.eks.us-east-1.amazonaws.com/id/${CLUSTER_OIDC}:sub": "system:serviceaccount:kube-system:efs-csi-controller-sa"
        }
      }
    }
  ]
}
EOT

aws iam create-role $(ppo) --role-name ${CLUSTER_NAME}_AmazonEKS_EFS_CSI_DriverRole --assume-role-policy-document file://trust-policy.json

aws iam attach-role-policy $(ppo) \
  --policy-arn arn:aws:iam::${AWS_ACCOUNT_ID}:policy/${CLUSTER_NAME}_AmazonEKS_EFS_CSI_Driver_Policy \
  --role-name ${CLUSTER_NAME}_AmazonEKS_EFS_CSI_DriverRole

cat <<EOT > efs-service-account.yaml
---
apiVersion: v1
kind: ServiceAccount
metadata:
  name: efs-csi-controller-sa
  namespace: kube-system
  labels:
    app.kubernetes.io/name: aws-efs-csi-driver
  annotations:
    eks.amazonaws.com/role-arn: arn:aws:iam::${AWS_ACCOUNT_ID}:role/${CLUSTER_NAME}_AmazonEKS_EFS_CSI_DriverRole
EOT

kubectl apply -f efs-service-account.yaml

helm repo add aws-efs-csi-driver https://kubernetes-sigs.github.io/aws-efs-csi-driver/
helm repo update

helm upgrade \
  -i aws-efs-csi-driver aws-efs-csi-driver/aws-efs-csi-driver \
  --namespace kube-system \
  --set image.repository=${CONTAINER_IMAGE_REGISTRY}/eks/aws-efs-csi-driver \
  --set controller.serviceAccount.create=false \
  --set controller.serviceAccount.name=efs-csi-controller-sa

kubectl get pod -n kube-system -l "app.kubernetes.io/name=aws-efs-csi-driver,app.kubernetes.io/instance=aws-efs-csi-driver"

EFS_ID=$(aws efs create-file-system --tags "Key=Name,Value=${CLUSTER_NAME}_file-system" $(pj)|jq -r .FileSystemId)
aws efs describe-file-systems --file-system-id $EFS_ID $(ppo)

fs_status="aws efs describe-file-systems --file-system-id $EFS_ID $(pj)|jq -r .FileSystems[0].LifeCycleState"
spintil "$fs_status" "available" "file system"

efs_posix_user="--posix-user Uid=1000,Gid=1000"
efs_root_dir="--root-directory Path=/storage,CreationInfo={OwnerUid=1000,OwnerGid=1000,Permissions=0755}"
EFS_AP_ID=$(aws efs create-access-point --file-system-id $EFS_ID $efs_posix_user $efs_root_dir $(pj)|jq -r .AccessPointId)
aws efs describe-access-points --access-point-id $EFS_AP_ID $(ppo)

fs_ap_status="aws efs describe-access-points --access-point-id $EFS_AP_ID $(pj)|jq -r '.AccessPoints[0].LifeCycleState'"
spintil "$fs_ap_status" "available" "file system access point"

cat <<EOT > storageclass.yaml
kind: StorageClass
apiVersion: storage.k8s.io/v1
metadata:
  name: efs-sc
provisioner: efs.csi.aws.com
EOT

kubectl apply -f storageclass.yaml
kubectl patch storageclass gp2 -p '{"metadata": {"annotations":{"storageclass.kubernetes.io/is-default-class":"false"}}}'
kubectl patch storageclass efs-sc -p '{"metadata": {"annotations":{"storageclass.kubernetes.io/is-default-class":"true"}}}'

cat <<EOT > pv.yaml
apiVersion: v1
kind: PersistentVolume
metadata:
  name: anaconda-storage
spec:
  capacity:
    storage: 800Gi
  volumeMode: Filesystem
  accessModes:
    - ReadWriteMany
  persistentVolumeReclaimPolicy: Retain
  storageClassName: efs-sc
  csi:
    driver: efs.csi.aws.com
    volumeHandle: ${EFS_ID}::${EFS_AP_ID}
---
apiVersion: v1
kind: PersistentVolume
metadata:
  name: anaconda-persistence
spec:
  capacity:
    storage: 40Gi
  volumeMode: Filesystem
  accessModes:
    - ReadWriteMany
  persistentVolumeReclaimPolicy: Retain
  storageClassName: efs-sc
  csi:
    driver: efs.csi.aws.com
    volumeHandle: ${EFS_ID}::${EFS_AP_ID}
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: anaconda-storage 
  namespace: anaconda-enterprise 
  labels:  
    app.kubernetes.io/name: anaconda-enterprise
    app.kubernetes.io/instance: anaconda-enterprise
    app.kubernetes.io/component: pvc 
  annotations:
    "helm.sh/hook": "pre-install"
    "helm.sh/hook-delete-policy": "before-hook-creation"
    "helm.sh/resource-policy": "keep"
spec:
  storageClassName: efs-sc
  volumeName: anaconda-storage
  accessModes:
  - ReadWriteMany
  resources:
    requests:
      storage: 800Gi
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: anaconda-persistence 
  namespace: anaconda-enterprise
  labels:  
    app.kubernetes.io/name: anaconda-enterprise
    app.kubernetes.io/instance: anaconda-enterprise
    app.kubernetes.io/component: pvc
  annotations:
    "helm.sh/hook": "pre-install"
    "helm.sh/hook-delete-policy": "before-hook-creation"
    "helm.sh/resource-policy": "keep"
spec:
  storageClassName: efs-sc
  volumeName: anaconda-persistence
  accessModes:
  - ReadWriteMany
  resources:
    requests:
      storage: 40Gi
EOT

kubectl apply -f pv.yaml

cat <<EOT > rbac.yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: anaconda-enterprise
  namespace: anaconda-enterprise
rules:
  - verbs:
      - get
      - list
    apiGroups:
      - ''
    resources:
      - namespaces
      - pods/log
      - events
  - verbs:
      - create
      - delete
      - get
      - list
      - patch
      - update
      - watch
    apiGroups:
      - ''
    resources:
      - configmaps
      - secrets
      - pods
      - persistentvolumeclaims
      - endpoints
      - services
  - verbs:
      - create
      - delete
      - get
      - list
      - patch
      - update
      - watch
    apiGroups:
      - apps
    resources:
      - deployments
      - replicasets
      - statefulsets
  - verbs:
      - create
      - delete
      - get
      - list
      - patch
      - update
      - watch
    apiGroups:
      - batch
    resources:
      - jobs
      - cronjobs
  - verbs:
      - create
      - delete
      - get
      - list
      - patch
      - update
      - watch
    apiGroups:
      - extensions
    resources:
      - deployments
      - replicasets
  - verbs:
      - create
      - delete
      - get
      - list
      - patch
      - update
      - watch
    apiGroups:
      - networking.k8s.io
    resources:
      - ingresses
  - verbs:
      - create
      - delete
      - get
      - list
      - patch
      - update
      - watch
    apiGroups:
      - route.openshift.io
    resources:
      - routes
      - routes/custom-host
  - verbs:
      - get
      - list
    apiGroups:
      - ''
    resources:
      - serviceaccounts
      - roles
---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: anaconda-enterprise
  namespace: anaconda-enterprise
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: Role
  name: anaconda-enterprise
subjects:
  - kind: ServiceAccount
    name: anaconda-enterprise
EOT

kubectl apply -f rbac.yaml

cat <<EOT > clusterrole.yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: anaconda-enterprise-ingress
rules:
  - verbs:
      - '*'
    apiGroups:
      - '*'
    resources:
      - ingressclasses
  - verbs:
      - patch
    apiGroups:
      - '*'
    resources:
      - events
  - verbs:
      - list
      - watch
    apiGroups:
      - '*'
    resources:
      - secrets
      - endpoints
      - ingresses
      - services
      - pods
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRoleBinding
metadata:
  name: anaconda-enterprise-ingress
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: ClusterRole
  name: anaconda-enterprise-ingress
subjects:
  - kind: ServiceAccount
    name: anaconda-enterprise
    namespace: anaconda-enterprise
EOT

kubectl apply -f clusterrole.yaml

cat <<EOT > anaconda-enterprise-sa.yaml
---
apiVersion: v1
kind: ServiceAccount
metadata:
  name: anaconda-enterprise
  namespace: anaconda-enterprise
EOT

kubectl apply -f anaconda-enterprise-sa.yaml

valuesyml=${HELM_CHART_DIR}/values.yaml
yq e -i ".hostname = \"${AE5_HOSTNAME}\"" $valuesyml
yq e -i '.ingress.install = true' $valuesyml
yq e -i '.ingress.installClass = true' $valuesyml
yq e -i '.image.pullSecrets = "**********"

kubectl create secret docker-registry regcred \
  --docker-server=https://index.docker.io/v1/ \
  --docker-username=aecustomers \
  --docker-password= "**********"

echo "Running helm install... "
helm install anaconda-enterprise -f $valuesyml ./${HELM_CHART_DIR}/Anaconda-Enterprise --output table
helm list
rprise --output table
helm list
