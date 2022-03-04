#date: 2022-03-04T16:56:06Z
#url: https://api.github.com/gists/80c61832ef2260b6915af4f5f760549c
#owner: https://api.github.com/users/prom3theu5

# Configuration
export PUBLIC_DNS=<public hostname>
export PUBLIC_IP=<public host IP>
export PRIVATE_IP=<private host IP>

mkdir docker-ca
chmod 0700 docker-ca/
cd docker-ca/

# CA key
openssl genrsa -aes256 -out ca-key.pem 2048
# CA certificate
openssl req -new -x509 -days 365 -key ca-key.pem -sha256 -out ca.pem

# Server key
openssl genrsa -out server-key.pem 2048
# Server CSR on DNS name
openssl req -subj "/CN==${PUBLIC_DNS}" -new -key server-key.pem -out server.csr
# Alts on IPs
echo "subjectAltName = IP:${PUBLIC_IP},IP:${PRIVATE_IP},IP:127.0.0.1" > extfile.cnf
# Server certificate
openssl x509 -req -days 365 -in server.csr -CA ca.pem -CAkey ca-key.pem -CAcreateserial -out server-cert.pem -extfile extfile.cnf

# Client key
openssl genrsa -out client-key.pem 2048
# Client CSR
openssl req -subj '/CN=client' -new -key client-key.pem -out client.csr
# clientAuth
echo extendedKeyUsage = clientAuth > extfile.cnf
# Client certificate
openssl x509 -req -days 365 -in client.csr -CA ca.pem -CAkey ca-key.pem -CAcreateserial -out client-cert.pem -extfile extfile.cnf

# Securing
chmod -v 0400 *-key.pem
chmod -v 0444 ca.pem *-cert.pem

# Moving
sudo mkdir -p /etc/docker
sudo chown root:docker /etc/docker
sudo chmod 700 /etc/docker
sudo cp ~/docker-ca/{ca,server-*}.pem /etc/docker
