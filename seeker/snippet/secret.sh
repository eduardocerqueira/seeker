#date: 2022-10-10T17:20:44Z
#url: https://api.github.com/gists/94a15e32ab01ee2168dbb6fa2086cb7e
#owner: https://api.github.com/users/adamancini

#!/bin/bash

set -e

bail() {
    printf "${RED}$1${NC}\n" 1>&2
    exit 1
}

function registry_pki_secret() {
    if [ -z "$DOCKER_REGISTRY_IP" ]; then
        bail "Docker registry address required"
    fi

    local tmp="/tmp/registry-pki"
    rm -rf "$tmp"
    mkdir -p "$tmp"
    pushd "$tmp"

    cat > registry.cnf <<EOF
[ req ]
default_bits = 2048
prompt = no
default_md = sha256
req_extensions = req_ext
distinguished_name = dn

[ dn ]
CN = registry.kurl.svc.cluster.local

[ req_ext ]
subjectAltName = @alt_names

[ v3_ext ]
authorityKeyIdentifier=keyid,issuer:always
basicConstraints=CA:FALSE
keyUsage=nonRepudiation,digitalSignature,keyEncipherment
extendedKeyUsage=serverAuth
subjectAltName=@alt_names

[ alt_names ]
DNS.1 = registry
DNS.2 = registry.kurl
DNS.3 = registry.kurl.svc
DNS.4 = registry.kurl.svc.cluster
DNS.5 = registry.kurl.svc.cluster.local
IP.1 = $DOCKER_REGISTRY_IP
EOF

    if [ -n "$REGISTRY_PUBLISH_PORT" ]; then
        echo "IP.2 = $PRIVATE_ADDRESS" >> registry.cnf

        if [ -n "$PUBLIC_ADDRESS" ]; then
            echo "IP.3 = $PUBLIC_ADDRESS" >> registry.cnf
        fi
    fi

    local ca_crt="/etc/kubernetes/pki/ca.crt"
    local ca_key="/etc/kubernetes/pki/ca.key"

    openssl req -newkey rsa:2048 -nodes -keyout registry.key -out registry.csr -config registry.cnf
    openssl x509 -req -days 365 -in registry.csr -CA "${ca_crt}" -CAkey "${ca_key}" -CAcreateserial -out registry.crt -extensions v3_ext -extfile registry.cnf

    # rotate the cert and restart the pod every time
    kubectl -n kurl delete secret registry-pki || true
    kubectl -n kurl create secret generic registry-pki --from-file= "**********"=registry.crt
    kubectl -n kurl delete pod -l app=registry || true

    popd
    rm -r "$tmp"
}

export KUBECONFIG=/etc/kubernetes/admin.conf
export DOCKER_REGISTRY_IP=$(kubectl -n kurl get svc registry -o jsonpath='{ .spec.clusterIP }')
registry_pki_secret
pki_secret
