#date: 2023-07-07T16:53:22Z
#url: https://api.github.com/gists/b584b04ae023a3e3b57dbedb9dec51ee
#owner: https://api.github.com/users/sgnconnects

# https://kubernetes.io/docs/concepts/services-networking/service/#external-ips
# If there are external IPs that route to one or more cluster nodes, Kubernetes services can be exposed on those externalIPs.
# Traffic that ingresses into the cluster with the external IP (as destination IP), on the service port, will be routed to
# one of the service endpoints.

# Create test app pods
cat <<EOF | kubectl create -f -
apiVersion: v1
kind: Pod
metadata:
  name: busybox-1
  labels:
    app: busybox-1
spec:
  containers:
  - name: busybox
    image: busybox
    args:
    - nc
    - "-lk"
    - "-v"
    - "-p"
    - "9111"
    - "-e"
    - echo
    - "hello from busybox-1"
---
apiVersion: v1
kind: Pod
metadata:
  name: busybox-2
  labels:
    app: busybox-2
spec:
  containers:
  - name: busybox
    image: busybox
    args:
    - nc
    - "-lk"
    - "-v"
    - "-p"
    - "9111"
    - "-e"
    - echo
    - "hello from busybox-2"
EOF

# Create services
cat <<EOF | kubectl create -f -
kind: Service
apiVersion: v1
metadata:
  name: service-1
spec:
  selector:
    app: busybox-1
  ports:
  - name: grpc
    protocol: TCP
    port: 9111
    targetPort: 9111
  externalIPs:
  - "10.10.0.1"
---
kind: Service
apiVersion: v1
metadata:
  name: service-2
spec:
  selector:
    app: busybox-2
  ports:
  - name: grpc
    protocol: TCP
    port: 9111
    targetPort: 9111
  externalIPs:
  - "10.10.0.2"
EOF
