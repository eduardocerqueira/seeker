#date: 2022-01-10T17:00:03Z
#url: https://api.github.com/gists/a8e2090839e6ec2b6286ee6515f1deba
#owner: https://api.github.com/users/bcollard

#!/bin/sh

export NS="gloo-system"
# helm
# download helm 3.6.3 from https://github.com/helm/helm/releases/tag/v3.6.3

# GLOO EDGE
helm repo update
helm upgrade -i gloo glooe/gloo-ee --namespace ${NS} --version 1.8.9 \
  --create-namespace --set-string license_key="$LICENSE_KEY" -f values.yaml

# HTTPBIN
kubectl apply -f httpbin.yaml

kubectl apply -f - <<EOF
apiVersion: gateway.solo.io/v1
kind: VirtualService
metadata:
  name: httpbin
  namespace: gloo-system
spec:
  virtualHost:
    domains:
    - '*'
    routes:
    - matchers:
      - prefix: /
      routeAction:
        single:
          upstream:
            name: default-httpbin-8000
            namespace: gloo-system
EOF


# ----------------------------------------------------------
# custom ext auth deployment
# will be the httpbin app, proxied by nginx with TLS on
# ----------------------------------------------------------

# 1 approved CA, signing backendA server cert
mkdir authorized-ca
cd authorized-ca
openssl genrsa -out authorized-ca.key 2048

# authorized CA
openssl req -x509 -new -nodes -sha512 -days 365 \
 -subj "/C=US/ST=Massachusetts/L=Boston/O=Solo-io/OU=pki/CN=authorized-ca" \
 -key authorized-ca.key \
 -out authorized-ca.crt

## server A cert signed by authrorized CA
openssl genrsa -out server-A.key 2048

openssl req -sha512 -new \
    -subj "/C=US/ST=Massachusetts/L=Boston/O=Solo-io/OU=pki/CN=server-A" \
    -key server-A.key \
    -out server-A.csr

cat > server.ext <<-EOF
keyUsage = critical,digitalSignature,keyEncipherment
basicConstraints = CA:FALSE
extendedKeyUsage = serverAuth
subjectKeyIdentifier = hash
EOF

openssl x509 -req -sha512 -days 365 \
    -extfile server.ext \
    -CA authorized-ca.crt -CAkey authorized-ca.key -CAcreateserial \
    -in server-A.csr \
    -out server-A.crt

openssl x509 -in server-A.crt -text

# nginx for server A (exposing a cert signed by an authorized CA)
cat server-A.crt authorized-ca.crt > chain.crt

# nginxsecret-server-a contains the server cert and the TLS key
kubectl create secret tls nginxsecret-server-a --cert=chain.crt --key server-A.key

# deploy nginx with the server cert (with full chain). It proxies to httpbin
k apply -f nginx-a.yaml

# ----------------------------------------------------------
# create a dedicated upstream name for custom auth
# with a sslConfig block
# ----------------------------------------------------------

# register the server cert as a trusted store for the Upstream object
glooctl create secret tls --rootca authorized-ca.crt authorized-ca

cd ..

kubectl apply -f - <<EOF
apiVersion: gloo.solo.io/v1
kind: Upstream
metadata:
  name: custom-auth-tls
  namespace: gloo-system
spec:
  discoveryMetadata: {}
  kube:
    selector:
      app: nginx-server-a
    serviceName: server-a
    serviceNamespace: default
    servicePort: 443
  sslConfig:
    secretRef:
      name: authorized-ca
      namespace: gloo-system
  healthChecks:
  - healthyThreshold: 1
    httpHealthCheck:
      path: /status/200
    interval: 5s
    timeout: 1s
    unhealthyThreshold: 3
EOF


# ----------------------------------------------------------
# configure settings with namedExtAuth
# ----------------------------------------------------------

# .Values.settings.kubeResourceOverride

kubectl -n gloo-system patch st/default --type merge -p '
spec:
  namedExtauth:
    extAuthBodyPassthrough:
      clearRouteCache: true
      extauthzServerRef:
        name: custom-auth-tls
        namespace: gloo-system
      httpService:
        pathPrefix: /
      requestBody:
        packAsBytes: false
      requestTimeout: 1s
      statusOnError: 403
'



# ----------------------------------------------------------
# add extauth to the default route in the VS
# ----------------------------------------------------------

kubectl apply -f - <<EOF
apiVersion: gateway.solo.io/v1
kind: VirtualService
metadata:
  name: httpbin
  namespace: gloo-system
spec:
  virtualHost:
    domains:
    - '*'
    routes:
    - matchers:
      - prefix: /
      routeAction:
        single:
          upstream:
            name: default-httpbin-8000
            namespace: gloo-system
      options:
        extauth:
          customAuth:
            name: extAuthBodyPassthrough
EOF

# ----------------------------------------------------------
# scale up gw-proxy to 3 replicas
# ----------------------------------------------------------
kgloo scale deploy/gateway-proxy --replicas 3



# ----------------------------------------------------------
# cert-manager
# ----------------------------------------------------------
kubectl create namespace cert-manager

helm repo add jetstack https://charts.jetstack.io

helm repo update

helm install \
  cert-manager jetstack/cert-manager \
  --namespace cert-manager \
  --version v1.5.4 \
  --set installCRDs=true


# CA secrets

## authorized CA
crt=`cat authorized-ca/authorized-ca.crt | base64 -w0`
key=`cat authorized-ca/authorized-ca.key | base64 -w0`

kubectl apply -f - << EOF
apiVersion: v1
kind: Secret
metadata:
  name: ca-authorized-with-key-4-cm
  namespace: default
data:
  tls.crt: $crt
  tls.key: $key
EOF

## unknown CA
crt=`cat unknown-ca/unknown-ca.crt | base64 -w0`
key=`cat unknown-ca/unknown-ca.key | base64 -w0`

kubectl apply -f - << EOF
apiVersion: v1
kind: Secret
metadata:
  name: ca-unknown-with-key-4-cm
  namespace: default
data:
  tls.crt: $crt
  tls.key: $key
EOF


## authorized CA issuer (cert-manager)
kubectl apply -f - << EOF
apiVersion: cert-manager.io/v1
kind: Issuer
metadata:
  name: authorized-ca-issuer
  namespace: default
spec:
  ca:
    secretName: ca-authorized-with-key-4-cm
EOF


## unknown CA issuer (cert-manager)
kubectl apply -f - << EOF
apiVersion: cert-manager.io/v1
kind: Issuer
metadata:
  name: unknown-ca-issuer
  namespace: default
spec:
  ca:
    secretName: ca-unknown-with-key-4-cm
EOF

## Certificate - authorized CA
kubectl apply -f - <<EOF
apiVersion: cert-manager.io/v1
kind: Certificate
metadata:
  name: server-cert-with-authorized-ca-by-cm
  namespace: default
spec:
  # Secret names are always required.
  secretName: server-cert-with-authorized-ca-by-cm
  duration: 2160h # 90d
  renewBefore: 360h # 15d
  isCA: false
  privateKey:
    algorithm: RSA
    encoding: PKCS1
    size: 2048
  usages:
    - server auth
  # At least one of a DNS Name, URI, or IP address is required.
  dnsNames:
    - mydomain.com
  # Issuer references are always required.
  issuerRef:
    name: authorized-ca-issuer
    # We can reference ClusterIssuers by changing the kind here.
    # The default value is Issuer (i.e. a locally namespaced Issuer)
    kind: Issuer
    # This is optional since cert-manager will default to this value however
    # if you are using an external issuer, change this to that issuer group.
    group: cert-manager.io
EOF

## Certificate - unknown CA
kubectl apply -f - <<EOF
apiVersion: cert-manager.io/v1
kind: Certificate
metadata:
  name: server-cert-with-unknown-ca-by-cm
  namespace: default
spec:
  # Secret names are always required.
  secretName: server-cert-with-unknown-ca-by-cm
  duration: 2160h # 90d
  renewBefore: 360h # 15d
  isCA: false
  privateKey:
    algorithm: RSA
    encoding: PKCS1
    size: 2048
  usages:
    - server auth
  # At least one of a DNS Name, URI, or IP address is required.
  dnsNames:
    - mydomain.com
  # Issuer references are always required.
  issuerRef:
    name: unknown-ca-issuer
    # We can reference ClusterIssuers by changing the kind here.
    # The default value is Issuer (i.e. a locally namespaced Issuer)
    kind: Issuer
    # This is optional since cert-manager will default to this value however
    # if you are using an external issuer, change this to that issuer group.
    group: cert-manager.io
EOF


# ----------------------------------------------------------
# prepare debug
# ----------------------------------------------------------
pkill kubectl

# port forward to grafana
kgloo port-forward svc/glooe-grafana 3000:80 &

# port-forward to the 3 replicas, HTTP listerner / Gateway
i=1; for g in $(kubectl -n gloo-system get po -l "gloo=gateway-proxy" -o jsonpath='{range .items[*]}{.metadata.name}{"\n"}{end}'); do kubectl -n gloo-system port-forward $g 808$i:8080 & ((i++)); done


# port-forward to the 3 replicas, envoy admin port
i=1; for g in $(kubectl -n gloo-system get po -l "gloo=gateway-proxy" -o jsonpath='{range .items[*]}{.metadata.name}{"\n"}{end}'); do kubectl -n gloo-system port-forward $g 1900$i:19000 & ((i++)); done


# ----------------------------------------------------------
# new test: switch to a secret generated by the authorized CA
# ----------------------------------------------------------

# revert to the correct / authorized CA trust store
kubectl apply -f - <<EOF
apiVersion: gloo.solo.io/v1
kind: Upstream
metadata:
  name: custom-auth-tls
  namespace: gloo-system
spec:
  discoveryMetadata: {}
  kube:
    selector:
      app: nginx-server-a
    serviceName: server-a
    serviceNamespace: default
    servicePort: 443
  sslConfig:
    secretRef:
      name: server-cert-with-authorized-ca-by-cm
      namespace: default
  healthChecks:
  - healthyThreshold: 1
    httpHealthCheck:
      path: /status/200
    interval: 5s
    timeout: 2s
    unhealthyThreshold: 3
EOF


#############
# Long-lived test
#############

(
  hey -c 1 -q 1 -z 30m -m GET -t 2 --disable-keepalive -cpus 1 http://127.0.0.1:8081/headers &
  hey -c 1 -q 1 -z 30m -m GET -t 2 --disable-keepalive -cpus 1 http://127.0.0.1:8082/headers &
  hey -c 1 -q 1 -z 30m -m GET -t 2 --disable-keepalive -cpus 1 http://127.0.0.1:8083/headers &
)

## check stats
curl -s 127.0.0.1:19001/stats | grep -i -E 'cluster.custom-auth-tls_gloo-system.health_check.network_failure|cluster.custom-auth-tls_gloo-system.ssl.fail_verify_error'
curl -s 127.0.0.1:19002/stats | grep -i -E 'cluster.custom-auth-tls_gloo-system.health_check.network_failure|cluster.custom-auth-tls_gloo-system.ssl.fail_verify_error'
curl -s 127.0.0.1:19003/stats | grep -i -E 'cluster.custom-auth-tls_gloo-system.health_check.network_failure|cluster.custom-auth-tls_gloo-system.ssl.fail_verify_error'
# 0 error

##################
# rotate !!
##################
k cert-manager renew server-cert-with-authorized-ca-by-cm

## check stats
curl -s 127.0.0.1:19001/stats | grep -i -E 'cluster.custom-auth-tls_gloo-system.health_check.network_failure|cluster.custom-auth-tls_gloo-system.ssl.fail_verify_error'
curl -s 127.0.0.1:19002/stats | grep -i -E 'cluster.custom-auth-tls_gloo-system.health_check.network_failure|cluster.custom-auth-tls_gloo-system.ssl.fail_verify_error'
curl -s 127.0.0.1:19003/stats | grep -i -E 'cluster.custom-auth-tls_gloo-system.health_check.network_failure|cluster.custom-auth-tls_gloo-system.ssl.fail_verify_error'
# 0 error
# still good


##################
# unauthorized CA
##################

kubectl apply -f - <<EOF
apiVersion: gloo.solo.io/v1
kind: Upstream
metadata:
  name: custom-auth-tls
  namespace: gloo-system
spec:
  discoveryMetadata: {}
  kube:
    selector:
      app: nginx-server-a
    serviceName: server-a
    serviceNamespace: default
    servicePort: 443
  sslConfig:
    secretRef:
      name: server-cert-with-unknown-ca-by-cm
      namespace: default
  healthChecks:
  - healthyThreshold: 1
    httpHealthCheck:
      path: /status/200
    interval: 5s
    timeout: 2s
    unhealthyThreshold: 3
EOF

curl -s 127.0.0.1:19001/stats | grep -i -E 'cluster.custom-auth-tls_gloo-system.health_check.network_failure|cluster.custom-auth-tls_gloo-system.ssl.fail_verify_error'
curl -s 127.0.0.1:19002/stats | grep -i -E 'cluster.custom-auth-tls_gloo-system.health_check.network_failure|cluster.custom-auth-tls_gloo-system.ssl.fail_verify_error'
curl -s 127.0.0.1:19003/stats | grep -i -E 'cluster.custom-auth-tls_gloo-system.health_check.network_failure|cluster.custom-auth-tls_gloo-system.ssl.fail_verify_error'
# errors on HC and SSL verify, which are legit


# revert to the correct / authorized CA trust store
kubectl apply -f - <<EOF
apiVersion: gloo.solo.io/v1
kind: Upstream
metadata:
  name: custom-auth-tls
  namespace: gloo-system
spec:
  discoveryMetadata: {}
  kube:
    selector:
      app: nginx-server-a
    serviceName: server-a
    serviceNamespace: default
    servicePort: 443
  sslConfig:
    secretRef:
      name: server-cert-with-authorized-ca-by-cm
      namespace: default
  healthChecks:
  - healthyThreshold: 1
    httpHealthCheck:
      path: /status/200
    interval: 5s
    timeout: 2s
    unhealthyThreshold: 3
EOF

pkill -INT hey
pkill kubectl









