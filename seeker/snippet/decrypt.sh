#date: 2023-11-06T16:48:07Z
#url: https://api.github.com/gists/d5f590d26a581ba46b8481357ac3bf5b
#owner: https://api.github.com/users/hanteed

#!/bin/bash
# based on http://sandilands.info/sgordon/public-key-encryption-and-digital-signatures-using-openssl

#Priv Key can be even ssh
PRIVKEY=private_key  #Can be link to ssh priv key: ~/.ssh/id_rsa
ENCRYPTED_FILE=test.txt.encrypted
DECRYPTED_FILE=test.txt.decrypted
ORIGINAL_TO_COMPARE=test.txt

set -x
#ssh-keygen -e -f "${PUBSSHKEY}" -m PKCS8 > "${PUBKEY}"
openssl pkeyutl -decrypt -inkey "${PRIVKEY}" -in "${ENCRYPTED_FILE}" -out "${DECRYPTED_FILE}"
cmp test.txt.decrypted test.txt && echo 'Decrypted is same as original'