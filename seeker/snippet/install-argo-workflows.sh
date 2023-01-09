#date: 2023-01-09T16:49:03Z
#url: https://api.github.com/gists/408e3b7874a98ef3f7adb87a45133d92
#owner: https://api.github.com/users/koosie0507

#!/bin/bash

ARGO_VERSION="${ARGO_VERSION:-"3.4.4"}"
ARGO_NAMESPACE="${ARGO_NAMESPACE:-"argo"}"
read -r -d '' ARGO_AUTH_MODE_SERVER_OP << 'EOF'
[
    {
        "op": "replace",
        "path": "/spec/template/spec/containers/0/args",
        "value": [
            "server",
            "--auth-mode=server"
        ]
    }
]
EOF

set -euo pipefail

kubectl create namespace "${ARGO_NAMESPACE}" --dry-run=client -o yaml | kubectl apply -f -
kubectl apply -n "${ARGO_NAMESPACE}" -f "https://github.com/argoproj/argo-workflows/releases/download/v${ARGO_VERSION}/install.yaml"
kubectl patch deployment argo-server --namespace "${ARGO_NAMESPACE}" --type="json" -p="${ARGO_AUTH_MODE_SERVER_OP}"
