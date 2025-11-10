#date: 2025-11-10T16:53:45Z
#url: https://api.github.com/gists/44caa409cdf7d796506a7a2e61a4a0d5
#owner: https://api.github.com/users/bcollard

# mk kind-create 1 bco-global

direnv allow

# vars
export KO_NAMESPACE="kong-system"
export PUBLIC_GW_NAMESPACE="kong-gw-public"
export PUBLIC_GW_NAME="gw-public"
export PRIVATE_GW_NAMESPACE="kong-gw-private"
export PRIVATE_GW_NAME="gw-private"
alias kks="kubectl -n ${KO_NAMESPACE} "

# NS
kubectl --context $CTX_GLOBAL create namespace ${PRIVATE_GW_NAMESPACE} || true
kubectl --context $CTX_GLOBAL create namespace ${PUBLIC_GW_NAMESPACE} || true


##########################
# Kong Operator 
##########################
# Gateway API
kubectl --context $CTX_GLOBAL apply -f https://github.com/kubernetes-sigs/gateway-api/releases/download/v1.3.0/standard-install.yaml

helm repo add kong https://charts.konghq.com
helm repo update

helm --kube-context $CTX_GLOBAL upgrade --install kong-operator kong/kong-operator -n ${KO_NAMESPACE} \
  --create-namespace \
  --set image.tag=${KONG_KO_VERSION} \
  --set env.ENABLE_CONTROLLER_KONNECT=true \
  --values - <<EOF
env:
  watch_namespace: ${PUBLIC_GW_NAMESPACE},${PRIVATE_GW_NAMESPACE}
EOF
  
# --set global.webhooks.options.certManager.enabled=true

# install KO
kubectl --context $CTX_GLOBAL -n ${KO_NAMESPACE} wait \
    --for=condition=Available=true --timeout=120s \
    deployment/kong-operator-kong-operator-controller-manager



##########################
# Kong Operator License
##########################
kubectl --context $CTX_GLOBAL -n ${KO_NAMESPACE} apply -f - <<EOF
apiVersion: configuration.konghq.com/v1alpha1
kind: KongLicense
metadata:
  name: kong-license
rawLicenseString: '$(cat ${KONG_LICENSE_FILE})'
EOF



########################
# Gateway public
########################
# Gateway API CRs
kubectl --context $CTX_GLOBAL apply -f - <<EOF
kind: GatewayConfiguration
apiVersion: gateway-operator.konghq.com/v2beta1
metadata:
  name: ${PUBLIC_GW_NAME}
  namespace: ${PUBLIC_GW_NAMESPACE}
spec:
  dataPlaneOptions:
    deployment:
      podTemplateSpec:
        spec:
          containers:
          - name: proxy
            image: kong/kong-gateway:${KONG_VERSION}
  controlPlaneOptions:
    watchNamespaces:
      type: own
EOF
# no pod created yet
kubectl --context $CTX_GLOBAL apply -f - <<EOF
kind: GatewayClass
apiVersion: gateway.networking.k8s.io/v1
metadata:
  name: ${PUBLIC_GW_NAME}
  namespace: ${PUBLIC_GW_NAMESPACE}
spec:
  controllerName: konghq.com/gateway-operator
  parametersRef:
    group: gateway-operator.konghq.com
    kind: GatewayConfiguration
    name: ${PUBLIC_GW_NAME}
    namespace: ${PUBLIC_GW_NAMESPACE}
---
kind: Gateway
apiVersion: gateway.networking.k8s.io/v1
metadata:
  name: ${PUBLIC_GW_NAME}
  namespace: ${PUBLIC_GW_NAMESPACE}
spec:
  gatewayClassName: ${PUBLIC_GW_NAME}
  listeners:
  - name: http
    protocol: HTTP
    port: 80
EOF

# Upstream Service
kubectl --context $CTX_GLOBAL apply -f https://developer.konghq.com/manifests/kic/echo-service.yaml -n ${PUBLIC_GW_NAMESPACE}

# HTTPRoute
kubectl --context $CTX_GLOBAL apply -f - <<EOF
kind: HTTPRoute
apiVersion: gateway.networking.k8s.io/v1
metadata:
  name: echo
  namespace: ${PUBLIC_GW_NAMESPACE}
spec:
  parentRefs:
    - group: gateway.networking.k8s.io
      kind: Gateway
      name: ${PUBLIC_GW_NAME}
  rules:
    - matches:
        - path:
            type: PathPrefix
            value: /echo
      backendRefs:
        - name: echo
          port: 1027
EOF

# Fetch proxy IP
export PUBLIC_GW_IP=$(kubectl get gateway ${PUBLIC_GW_NAME} -n ${PUBLIC_GW_NAMESPACE} -o jsonpath='{.status.addresses[0].value}')

curl  "$PUBLIC_GW_IP/echo" \
  --no-progress-meter --fail-with-body 



########################
# Gateway private
########################
# Gateway API CRs
kubectl --context $CTX_GLOBAL apply -f - <<EOF
kind: GatewayConfiguration
apiVersion: gateway-operator.konghq.com/v2beta1
metadata:
  name: ${PRIVATE_GW_NAME}
  namespace: ${PRIVATE_GW_NAMESPACE}
spec:
  dataPlaneOptions:
    deployment:
      podTemplateSpec:
        spec:
          containers:
          - name: proxy
            image: kong/kong-gateway:${KONG_VERSION}
  controlPlaneOptions:
    watchNamespaces:
      type: own
EOF
# no pod created yet
kubectl --context $CTX_GLOBAL apply -f - <<EOF
kind: GatewayClass
apiVersion: gateway.networking.k8s.io/v1
metadata:
  name: ${PRIVATE_GW_NAME}
  namespace: ${PRIVATE_GW_NAMESPACE}
spec:
  controllerName: konghq.com/gateway-operator
  parametersRef:
    group: gateway-operator.konghq.com
    kind: GatewayConfiguration
    name: ${PRIVATE_GW_NAME}
    namespace: ${PRIVATE_GW_NAMESPACE}
---
kind: Gateway
apiVersion: gateway.networking.k8s.io/v1
metadata:
  name: ${PRIVATE_GW_NAME}
  namespace: ${PRIVATE_GW_NAMESPACE}
spec:
  gatewayClassName: ${PRIVATE_GW_NAME}
  listeners:
  - name: http
    protocol: HTTP
    port: 80
EOF

# Upstream Service
kubectl --context $CTX_GLOBAL apply -f https://developer.konghq.com/manifests/kic/echo-service.yaml -n ${PRIVATE_GW_NAMESPACE}


# HTTPRoute
kubectl --context $CTX_GLOBAL apply -f - <<EOF
kind: HTTPRoute
apiVersion: gateway.networking.k8s.io/v1
metadata:
  name: echo
  namespace: ${PRIVATE_GW_NAMESPACE}
spec:
  parentRefs:
    - group: gateway.networking.k8s.io
      kind: Gateway
      name: ${PRIVATE_GW_NAME}
  rules:
    - matches:
        - path:
            type: PathPrefix
            value: /echo
      backendRefs:
        - name: echo
          port: 1027
EOF

# Fetch proxy IP
export PRIVATE_GW_IP=$(kubectl get gateway ${PRIVATE_GW_NAME} -n ${PRIVATE_GW_NAMESPACE} -o jsonpath='{.status.addresses[0].value}')

curl  "$PRIVATE_GW_IP/echo" \
  --no-progress-meter --fail-with-body



