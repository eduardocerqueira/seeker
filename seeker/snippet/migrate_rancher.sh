#date: 2025-01-07T17:09:38Z
#url: https://api.github.com/gists/ce9e443279f9c87ae86be6aa266101ad
#owner: https://api.github.com/users/bvaliev

#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# Rancher migration script from an old RKE cluster to a new cluster (e.g., RKE2).
# Requirements:
# 1) Installed kubectl, helm, access to contexts in both clusters
# 2) The rancher-backup operator is installed in both clusters (or will be installed)
# 3) StorageLocation configured (S3/MinIO/mounting) to store Rancher backups
#
# Usage:
#   export KUBECONFIG_OLD=~/.kube/config-old
#   export KUBECONFIG_NEW=~/.kube/config-new
#   export BACKUP_NAME=rancher-backup-2025-01-07
#   export BACKUP_STORAGE_LOCATION=default-s3
#   export RESTORE_NAME=rancher-restore-2025-01-07
#   export RANCHER_CHART_VERSION="latest"  # or a specific version, e.g. “v2.7.5”
#   export RANCHER_NAMESPACE="cattle-system"
#   ./migrate_rancher.sh
###############################################################################

# --- Checking mandatory environment variables ---
: "${KUBECONFIG_OLD:?Variable KUBECONFIG_OLD is not set or empty}"
: "${KUBECONFIG_NEW:?Variable KUBECONFIG_NEW is not set or empty}"
: "${BACKUP_NAME:?Variable BACKUP_NAME is not set or empty}"
: "${BACKUP_STORAGE_LOCATION:?Variable BACKUP_STORAGE_LOCATION is not set or empty}"
: "${RESTORE_NAME:?Variable RESTORE_NAME is not set or empty}"
: "${RANCHER_CHART_VERSION:?Variable RANCHER_CHART_VERSION is not set or empty}"
: "${RANCHER_NAMESPACE:?Variable RANCHER_NAMESPACE is not set or empty}"

echo "=== Rancher Migration Script ==="
echo "Using old cluster kubeconfig: $KUBECONFIG_OLD"
echo "Using new cluster kubeconfig: $KUBECONFIG_NEW"
echo "Backup name: $BACKUP_NAME"
echo "Backup storage location: $BACKUP_STORAGE_LOCATION"
echo "Restore name: $RESTORE_NAME"
echo "Rancher chart version: $RANCHER_CHART_VERSION"
echo "Rancher namespace: $RANCHER_NAMESPACE"
echo "================================"

###############################################################################
# 1. Checking that the rancher-backup operator is installed in the old cluster.
###############################################################################
echo "[1/7] Checking/Installing Rancher Backup Operator in the old cluster..."
kubectl --kubeconfig "$KUBECONFIG_OLD" create ns cattle-resources 2>/dev/null || true

# You can use helm-chart rancher/backup-restore-operator or kubectl apply CRDs
# Below is an example via Helm
helm repo add rancher-latest https://releases.rancher.com/server-charts/latest 2>/dev/null || true
helm repo update

# Set/update the operator in namespace cattle-resources:
helm upgrade --install rancher-backup-operator \
  rancher-latest/backup-restore-operator \
  --namespace cattle-resources \
  --kubeconfig "$KUBECONFIG_OLD" \
  --wait

echo "[1/7] Rancher Backup Operator is ready in old cluster."
echo "-------------------------------------------------------"

###############################################################################
# 2. Creating Backup custom resource in old cluster
###############################################################################
echo "[2/7] Creating Backup custom resource in old cluster..."

cat <<EOF | kubectl --kubeconfig "$KUBECONFIG_OLD" apply -f -
apiVersion: resources.cattle.io/v1
kind: Backup
metadata:
  name: ${BACKUP_NAME}
  namespace: cattle-resources
spec:
  backupSource:
    rancherSecretMigrationVersion: "**********"
  storageLocation:
    name: ${BACKUP_STORAGE_LOCATION}
EOF

echo "Backup resource '${BACKUP_NAME}' created. Waiting for completion..."

status=""
while true; do
  status="$(kubectl --kubeconfig "$KUBECONFIG_OLD" -n cattle-resources get backup "${BACKUP_NAME}" -o jsonpath='{.status.phase}' || echo "")"
  if [[ "$status" == "Completed" ]]; then
    echo "Backup '${BACKUP_NAME}' completed successfully."
    break
  elif [[ "$status" == "Error" ]]; then
    echo "ERROR: Backup '${BACKUP_NAME}' ended with Error status!"
    exit 1
  else
    echo "   Backup phase: $status. Retrying in 10s..."
    sleep 10
  fi
done

echo "[2/7] Backup finished successfully."
echo "-------------------------------------------------------"

###############################################################################
# 3. (Optional) Download backup locally
# If you are using S3/MinIO/NFS, you can skip it,
# or vice versa, specify here the logic of downloading from S3.
###############################################################################
echo “[3/7] (Optional) Downloading backup artifact...”
# Example if you just want to get tar.gz from CR (only with localStorage):
# kubectl --kubeconfig "$KUBECONFIG_OLD" -n cattle-resources get backup "${BACKUP_NAME}" -o jsonpath='{.status.filename}' > /tmp/backup_filename.txt
# filename=$(cat /tmp/backup_filename.txt)
# kubectl --kubeconfig "$KUBECONFIG_OLD" -n cattle-resources cp "cattle-resources/$filename" "./${BACKUP_NAME}.tar.gz"

echo "Skipping actual download, assuming S3/MinIO storage is used."
echo "-------------------------------------------------------"

###############################################################################
# 4. Installation of Rancher (basic) and rancher-backup operator in a new cluster
###############################################################################
echo "[4/7] Installing rancher-backup operator in the new cluster..."

kubectl --kubeconfig "$KUBECONFIG_NEW" create ns cattle-resources 2>/dev/null || true

helm upgrade --install rancher-backup-operator \
  rancher-latest/backup-restore-operator \
  --namespace cattle-resources \
  --kubeconfig "$KUBECONFIG_NEW" \
  --wait

echo "[4/7] Rancher Backup Operator is ready in the new cluster."
echo "-------------------------------------------------------"

echo "[4a/7] Installing (or upgrading) Rancher in the new cluster..."

kubectl --kubeconfig "$KUBECONFIG_NEW" create ns "$RANCHER_NAMESPACE" 2>/dev/null || true

# Example of Rancher installation (without external DB). If necessary, add/modify
# parameters (hostname, ingress, tls, etc.).
helm upgrade --install rancher \
  rancher-latest/rancher \
  --namespace "$RANCHER_NAMESPACE" \
  --kubeconfig "$KUBECONFIG_NEW" \
  --version "$RANCHER_CHART_VERSION" \
  --set replicas=1 \
  --set hostname=rancher.local \
  --wait

echo "[4a/7] Rancher chart installed/upgraded in new cluster."
echo "-------------------------------------------------------"

###############################################################################
# 5. “Import” the backup to the new storage (if necessary)
# If you have S3/MinIO - it is enough that BackupStorageLocation points to the same
# bucket, and the Backup will be available.
###############################################################################
echo "[5/7] Verifying BackupStorageLocation in the new cluster..."

# Suppose we also have BACKUP_STORAGE_LOCATION = default-s3,
# and configured for the same buckets. If you need it differently, you can apply CR:
# cat <<EOF | kubectl --kubeconfig "$KUBECONFIG_NEW" apply -f -
# apiVersion: resources.cattle.io/v1
# kind: BackupStorageLocation
# metadata:
#   name: default-s3
#   namespace: cattle-resources
# spec:
#   # ...
# EOF

echo "Assuming the same S3/MinIO config is used. Make sure it's set up correctly."
echo "-------------------------------------------------------"

###############################################################################
# 6. Creating Restore resource in the new cluster
###############################################################################
echo "[6/7] Creating Restore resource in the new cluster..."

cat <<EOF | kubectl --kubeconfig "$KUBECONFIG_NEW" apply -f -
apiVersion: resources.cattle.io/v1
kind: Restore
metadata:
  name: ${RESTORE_NAME}
  namespace: cattle-resources
spec:
  backupName: ${BACKUP_NAME}
  storageLocation:
    name: ${BACKUP_STORAGE_LOCATION}
EOF

echo "Restore resource '${RESTORE_NAME}' created. Waiting for completion..."

status=""
while true; do
  status="$(kubectl --kubeconfig "$KUBECONFIG_NEW" -n cattle-resources get restore "${RESTORE_NAME}" -o jsonpath='{.status.phase}' || echo "")"
  if [[ "$status" == "Completed" ]]; then
    echo "Restore '${RESTORE_NAME}' completed successfully."
    break
  elif [[ "$status" == "Error" ]]; then
    echo "ERROR: Restore '${RESTORE_NAME}' ended with Error status!"
    exit 1
  else
    echo "   Restore phase: $status. Retrying in 10s..."
    sleep 10
  fi
done

echo "[6/7] Restore finished successfully."
echo "-------------------------------------------------------"

###############################################################################
# 7. Checking Rancher in the new cluster
###############################################################################
echo "[7/7] Checking Rancher in the new cluster..."

kubectl --kubeconfig "$KUBECONFIG_NEW" -n "$RANCHER_NAMESPACE" rollout status deploy/rancher --timeout=600s
echo "Rancher deployment is ready. Migration steps completed."

echo "-------------------------------------------------------"
echo “Verify that Rancher is now running on the new cluster.”
echo “You can go to the UI (hostname=rancher.local) or by NodePort/ingress.”
echo “Once everything is verified, you can delete the old cluster.”
echo "=== Migration script completed successfully ==="
