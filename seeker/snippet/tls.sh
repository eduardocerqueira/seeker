#date: 2022-05-17T17:15:01Z
#url: https://api.github.com/gists/4ae8b697550554324e54c2d3b687ff88
#owner: https://api.github.com/users/jacobmammoliti

# Generate the CA private key
openssl genrsa -out ca-key.pem 4096

# Create a configuration file for the CA certificate
cat <<EOF > ca_cert_config.txt
[req]
distinguished_name = req_distinguished_name
x509_extensions    = v3_ca
prompt             = no

[req_distinguished_name]
countryName             = CA
stateOrProvinceName     = Ontario
localityName            = Toronto
organizationName        = HashiCorp
commonName              = HashiCorp

[v3_ca]
basicConstraints        = critical,CA:TRUE
subjectKeyIdentifier    = hash
authorityKeyIdentifier  = keyid:always,issuer
EOF

# Generate a CA valid for 10 years
openssl req -new -x509 -days 3650 \
-config ca_cert_config.txt \
-key ca-key.pem \
-out ca.pem

# Generate a private key for the client certificate
openssl genrsa -out cert-key.pem 4096

# Create a configuration file for the client certificate
cat <<EOF > server_cert_config.txt
default_bit        = 4096
distinguished_name = req_distinguished_name
prompt             = no

[req_distinguished_name]
countryName             = CA
stateOrProvinceName     = Ontario
localityName            = Toronto
organizationName        = HashiCorp
commonName              = vault.hashicorp.com
EOF

# Create an extension and SAN file for the client certificate
# Add any additional SANs necessary for the Vault nodes
cat <<EOF > server_ext_config.txt
authorityKeyIdentifier = keyid,issuer
basicConstraints       = CA:FALSE
keyUsage               = digitalSignature, nonRepudiation, keyEncipherment, dataEncipherment
extendedKeyUsage       = serverAuth, clientAuth
subjectAltName         = @alt_names

[alt_names]
DNS.1 = vault-0.vault-internal
DNS.2 = vault-1.vault-internal
DNS.3 = vault-2.vault-internal
DNS.4 = vault-3.vault-internal
DNS.5 = vault-4.vault-internal
DNS.6 = vault.vault
DNS.7 = vault-active.vault
DNS.8 = vault-internal.vault
DNS.9 = vault.hashicorp.com
EOF

# Generate the Certificate Signing Request
openssl req -new -key cert-key.pem -out cert-csr.pem -config server_cert_config.txt

# Generate the signed certificate valid for 1 year
openssl x509 -req -in cert-csr.pem -out cert.pem \
-CA ca.pem -CAkey ca-key.pem -CAcreateserial \
-days 365 -sha512 -extfile server_ext_config.txt