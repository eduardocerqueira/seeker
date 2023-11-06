#date: 2023-11-06T16:48:07Z
#url: https://api.github.com/gists/d5f590d26a581ba46b8481357ac3bf5b
#owner: https://api.github.com/users/hanteed

#!/bin/bash
# based on http://sandilands.info/sgordon/public-key-encryption-and-digital-signatures-using-openssl
PUBSSHKEY=pub_ssh_key # can be link to ssh public key e.g.  ~/.ssh/id_rsa.pub
PUBKEY=pub.pkcs8
FILE_TO_ENCRYPT=test.txt 
ENCRYPTED_FILE=test.txt.encrypted
set -x
ssh-keygen -e -f "${PUBSSHKEY}" -m PKCS8 > "${PUBKEY}"
openssl pkeyutl -encrypt -pubin -inkey "${PUBKEY}" -in "${FILE_TO_ENCRYPT}" -out "${ENCRYPTED_FILE}"