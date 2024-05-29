#date: 2024-05-29T16:46:36Z
#url: https://api.github.com/gists/31b14114afaf8568fb6112f601d637b6
#owner: https://api.github.com/users/MoOyeg

#!/bin/bash 

cat <<EOF | oc apply -f -
apiVersion: objectbucket.io/v1alpha1
kind: ObjectBucketClaim
metadata:
  name: loki-bucket-odf
  namespace: openshift-logging 
spec:
  storageClassName: openshift-storage.noobaa.io
  generateBucketName: loki-bucket-odf
EOF

BUCKET_HOST=$(oc get -n openshift-logging configmap loki-bucket-odf -o jsonpath='{.data.BUCKET_HOST}')
BUCKET_NAME=$(oc get -n openshift-logging configmap loki-bucket-odf -o jsonpath='{.data.BUCKET_NAME}')
BUCKET_PORT=$(oc get -n openshift-logging configmap loki-bucket-odf -o jsonpath='{.data.BUCKET_PORT}')


ACCESS_KEY_ID= "**********"='{.data.AWS_ACCESS_KEY_ID}' | base64 -d)
SECRET_ACCESS_KEY= "**********"='{.data.AWS_SECRET_ACCESS_KEY}' | base64 -d)

oc create -n openshift-logging secret generic logging-loki-odf \
--from-literal= "**********"="${ACCESS_KEY_ID}" \
--from-literal= "**********"="${SECRET_ACCESS_KEY}" \
--from-literal=bucketnames="${BUCKET_NAME}" \
--from-literal=endpoint="https://${BUCKET_HOST}:${BUCKET_PORT}"

cat << EOF| oc apply -f -
apiVersion: loki.grafana.com/v1
kind: LokiStack
metadata:
  name: logging-loki 
  namespace: openshift-logging
spec:
  limits:
   global: 
      retention: 
        days: 1
  size: 1x.extra-small
  storage:
    schemas:
      - effectiveDate: '2023-10-15'
        version: v13
 "**********"  "**********"  "**********"  "**********"  "**********"s "**********"e "**********"c "**********"r "**********"e "**********"t "**********": "**********"
      name: logging-loki-odf
      type: s3 
  storageClassName: ocs-storagecluster-ceph-rbd 
  tenants:
    mode: openshift-logging
  managementState: Managed
EOF

oc patch lokistack/logging-loki -n openshift-logging --type merge -p '{"spec":{"storage":{"tls":{"caName":"openshift-service-ca.crt"}}}}'
#In one of my tests I still got an x509 error and had to delete logging-loki-ca-bundle and allow it be recreated to clear it

cat << EOF| oc apply -f -
apiVersion: logging.openshift.io/v1
kind: ClusterLogging
metadata:
  name: instance
  namespace: openshift-logging
spec:
  collection:
    type: vector
  logStore:
    lokistack:
      name: logging-loki
    retentionPolicy:
      application:
        maxAge: 1d
      audit:
        maxAge: 1d
      infra:
        maxAge: 1d
    type: lokistack
  visualization:
    type: ocp-console
  managementState: Managed
EOF