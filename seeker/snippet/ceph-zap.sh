#date: 2025-09-16T16:47:09Z
#url: https://api.github.com/gists/4e95a5cebc0b831b9e57b77bc3bc7ef4
#owner: https://api.github.com/users/sarahbx

#!/usr/bin/env bash
set -e

readonly WIPE_DISK=$1

cat <<EOF | oc create -f -
---
kind: Namespace
apiVersion: v1
metadata:
  name: ceph-zap
  labels:
    openshift.io/cluster-monitoring: "true"
    pod-security.kubernetes.io/enforce: privileged
    pod-security.kubernetes.io/audit: privileged
    pod-security.kubernetes.io/warn: privileged
    security.openshift.io/scc.podSecurityLabelSync: "false"
EOF

sleep 5

cat <<EOF | oc create -f -
---
apiVersion: v1
kind: ServiceAccount
metadata:
  name: ceph-zap-sa
  namespace: ceph-zap
---
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: ceph-zap-role
  namespace: ceph-zap
rules:
- apiGroups: [""]
  resources: ["pods", "nodes"]
  verbs: ["get", "list", "watch"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: ceph-zap-rb
  namespace: ceph-zap
subjects:
- kind: ServiceAccount
  name: ceph-zap-sa
  namespace: ceph-zap
roleRef:
  kind: Role
  name: ceph-zap-role
  apiGroup: rbac.authorization.k8s.io
EOF

oc adm policy add-scc-to-user privileged -z ceph-zap-sa -n ceph-zap

for _NODE in $(oc get nodes --selector='node-role.kubernetes.io/worker' --no-headers | awk '{print $1;}'); do
cat <<EOF | oc create -f -
---
apiVersion: batch/v1
kind: Job
metadata:
  name: ceph-zap-device-job-$_NODE
  namespace: ceph-zap
spec:
  template:
    metadata:
      labels:
        app: ceph-zap-device
    spec:
      serviceAccountName: ceph-zap-sa
      nodeSelector:
        kubernetes.io/hostname: $_NODE
      tolerations:
      - effect: NoSchedule
        key: node-role.kubernetes.io/master
        operator: Exists
      containers:
      - name: ceph-bluestore-tool
        image: quay.io/ceph/ceph:v19
        securityContext:
          privileged: true
        volumeMounts:
        - name: dev-disk
          mountPath: /dev/disk/by-path
        command: ["/bin/sh", "-c"]
        args:
          - "ceph-bluestore-tool zap-device --dev ${WIPE_DISK} --yes-i-really-really-mean-it"
      volumes:
      - name: dev-disk
        hostPath:
          path: /dev/disk/by-path
          type: Directory
      restartPolicy: Never
EOF
done

for job in $(oc get jobs -n ceph-zap -o custom-columns=NAME:.metadata.name --no-headers=true); do
    echo "Waiting for job $job to complete..."
    oc wait --for=condition=complete "job/$job" -n ceph-zap --timeout=60s
done

oc delete ns ceph-zap
