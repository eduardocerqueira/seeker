#date: 2024-05-29T16:45:16Z
#url: https://api.github.com/gists/6f743f8188239c835dbfc51c90019e5c
#owner: https://api.github.com/users/MoOyeg

#!/bin/bash 

oc new-project netobserv

cat <<EOF | oc apply -f -
apiVersion: objectbucket.io/v1alpha1
kind: ObjectBucketClaim
metadata:
  name: loki-net-bucket-odf
  namespace: netobserv 
spec:
  storageClassName: openshift-storage.noobaa.io
  generateBucketName: loki-net-bucket-odf
EOF

BUCKET_HOST=$(oc get -n netobserv configmap loki-net-bucket-odf -o jsonpath='{.data.BUCKET_HOST}')
BUCKET_NAME=$(oc get -n netobserv configmap loki-net-bucket-odf -o jsonpath='{.data.BUCKET_NAME}')
BUCKET_PORT=$(oc get -n netobserv configmap loki-net-bucket-odf -o jsonpath='{.data.BUCKET_PORT}')


ACCESS_KEY_ID= "**********"='{.data.AWS_ACCESS_KEY_ID}' | base64 -d)
SECRET_ACCESS_KEY= "**********"='{.data.AWS_SECRET_ACCESS_KEY}' | base64 -d)

oc create -n netobserv secret generic netobserv-loki-odf \
--from-literal= "**********"="${ACCESS_KEY_ID}" \
--from-literal= "**********"="${SECRET_ACCESS_KEY}" \
--from-literal=bucketnames="${BUCKET_NAME}" \
--from-literal=endpoint="https://${BUCKET_HOST}:${BUCKET_PORT}"

cat << EOF| oc apply -f -
apiVersion: loki.grafana.com/v1
kind: LokiStack
metadata:
  name: netobserv-loki 
  namespace: netobserv
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
      name: netobserv-loki-odf
      type: s3 
  storageClassName: ocs-storagecluster-ceph-rbd 
  tenants:
    mode: openshift-network
  managementState: Managed
EOF

oc patch lokistack/netobserv-loki -n netobserv --type merge -p '{"spec":{"storage":{"tls":{"caName":"openshift-service-ca.crt"}}}}'
#In one of my tests I still got an x509 error and had to delete logging-loki-ca-bundle and allow it be recreated to clear it

cat << EOF| oc apply -f -
apiVersion: flows.netobserv.io/v1beta2
kind: FlowCollector
metadata:
  name: cluster
spec:
  agent:
    ebpf:
      logLevel: info
      cacheMaxFlows: 100000
      sampling: 50
      imagePullPolicy: IfNotPresent
      excludeInterfaces:
        - lo
      kafkaBatchSize: 1048576
      cacheActiveTimeout: 5s
    ipfix:
      cacheActiveTimeout: 20s
      cacheMaxFlows: 400
      clusterNetworkOperator:
        namespace: openshift-network-operator
      forceSampleAll: false
      ovnKubernetes:
        containerName: ovnkube-node
        daemonSetName: ovnkube-node
        namespace: ovn-kubernetes
      sampling: 400
    type: eBPF
  consolePlugin:
    logLevel: info
    advanced:
      port: 9001
      register: true
    enable: true
    portNaming:
      enable: true
    quickFilters:
      - default: true
        filter:
          flow_layer: app
        name: Applications
      - filter:
          flow_layer: infra
        name: Infrastructure
      - default: true
        filter:
          dst_kind: Pod
          src_kind: Pod
        name: Pods network
      - filter:
          dst_kind: Service
        name: Services network
    imagePullPolicy: IfNotPresent
    autoscaler:
      maxReplicas: 3
      status: Disabled
    replicas: 1
  deploymentModel: Direct
  kafka:
    sasl:
      type: Disabled
    tls:
      enable: false
      insecureSkipVerify: false
  loki:
    advanced:
      writeMaxBackoff: 5s
      writeMaxRetries: 2
      writeMinBackoff: 1s
    writeTimeout: 10s
    microservices:
      ingesterUrl: 'http://loki-distributor:3100/'
      querierUrl: 'http://loki-query-frontend:3100/'
      tenantID: netobserv
      tls:
        enable: false
        insecureSkipVerify: false
    enable: true
    mode: LokiStack
    manual:
      authToken: "**********"
      ingesterUrl: 'http://loki:3100/'
      querierUrl: 'http://loki:3100/'
      statusTls:
        enable: false
        insecureSkipVerify: false
      tenantID: netobserv
      tls:
        enable: false
        insecureSkipVerify: false
    lokiStack:
      name: netobserv-loki
      namespace: netobserv
    readTimeout: 30s
    monolithic:
      tenantID: netobserv
      tls:
        enable: false
        insecureSkipVerify: false
      url: 'http://loki:3100/'
    writeBatchWait: 1s
    writeBatchSize: 102400
  namespace: netobserv
  processor:
    logLevel: info
    advanced:
      port: 2055
      conversationTerminatingTimeout: 5s
      conversationEndTimeout: 10s
      profilePort: 6060
      enableKubeProbes: true
      healthPort: 8080
      dropUnusedFields: true
      conversationHeartbeatInterval: 30s
    metrics:
      server:
        port: 9102
        tls:
          insecureSkipVerify: false
          type: Disabled
    multiClusterDeployment: false
    kafkaConsumerQueueCapacity: 1000
    imagePullPolicy: IfNotPresent
    kafkaConsumerAutoscaler:
      maxReplicas: 3
      status: Disabled
    logTypes: Flows
    kafkaConsumerReplicas: 3
    kafkaConsumerBatchSize: 10485760
EOF
