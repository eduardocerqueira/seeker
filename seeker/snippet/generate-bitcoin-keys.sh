#date: 2024-06-04T16:42:57Z
#url: https://api.github.com/gists/dc9141cdf966c6a2f52d2aa5a8ad5030
#owner: https://api.github.com/users/jean-charles

#!/bin/sh

PRIVATE_KEY="ECDSA"
PUBLIC_KEY="ECDSA.pub"
BITCOIN_PRIVATE_KEY="bitcoin"
BITCOIN_PUBLIC_KEY="bitcoin.pub"

echo "Generating private key"
openssl ecparam -genkey -name secp256k1 -rand /dev/random -out $PRIVATE_KEY

echo "Generating public key"
openssl ec -in $PRIVATE_KEY -pubout -out $PUBLIC_KEY

echo "Generating Bitcoin private key"
openssl ec -in $PRIVATE_KEY -outform DER|tail -c +8|head -c 32|xxd -p -c 32 > $BITCOIN_PRIVATE_KEY

echo "Generating Bitcoin public key"
openssl ec -in $PRIVATE_KEY -pubout -outform DER|tail -c 65|xxd -p -c 65 > $BITCOIN_PUBLIC_KEY