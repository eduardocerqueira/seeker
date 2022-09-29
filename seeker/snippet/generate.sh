#date: 2022-09-29T17:33:01Z
#url: https://api.github.com/gists/2b563526f3446773b87747873d6c5f8f
#owner: https://api.github.com/users/nausixkiz

#!/bin/sh

generate() {
    # Create private key for the server
    openssl genrsa -passout pass: "**********"
    # Remove passphrase
    openssl rsa -passin pass: "**********"
    # Create CSR for the server
    openssl req -new \
        -subj "/C=VN/ST=Mien Bac/L=Ha Noi/O=Ryo Software Ltd/OU=RyoSoft/CN=${2}" \
        -key server.key \
        -out server.csr
}

generate {password} {domain}     -out server.csr
}

generate {password} {domain}