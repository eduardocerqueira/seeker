#date: 2024-04-26T17:09:57Z
#url: https://api.github.com/gists/f382f779c0405a383e823563cc6013e0
#owner: https://api.github.com/users/sandip-unvested

#!/bin/bash
## Requires openssl, nodejs, jq
## FIXES for jwt.io compliance
# use base64Url encoding
# use echo -n in pack function
header='{
  "kid": "12345",
  "alg": "RS256"
}'
payload='{
  "iss": "<add here>",
  "sub": "<add here>",
  "jti": "unvested-test-1",
  "aud": "**********"://identity.moneyhub.co.uk/oidc/token",
  "iat": 1714143145,
  "exp": 1722009145,
}'

function pack () {
  # Remove line breaks and spaces
  echo $1 | sed -e "s/[\r\n]\+//g" | sed -e "s/ //g"
  }

function base64url_encode {
  (if [ -z "$1" ]; then cat -; else echo -n "$1"; fi) |
    openssl base64 -e -A |
      sed s/\\+/-/g |
      sed s/\\//_/g |
      sed -E s/=+$//
}

# just for debugging
function base64url_decode {
  INPUT=$(if [ -z "$1" ]; then echo -n $(cat -); else echo -n "$1"; fi)
  MOD=$(($(echo -n "$INPUT" | wc -c) % 4))
  PADDING=$(if [ $MOD -eq 2 ]; then echo -n '=='; elif [ $MOD -eq 3 ]; then echo -n '=' ; fi)
  echo -n "$INPUT$PADDING" |
    sed s/-/+/g |
    sed s/_/\\//g |
    openssl base64 -d -A
}

if [ ! -f private-key.pem ]; then
  # Private and Public keys
  openssl genrsa 2048 > private-key.pem
  openssl rsa -in private-key.pem -pubout -out public-key.pem
fi

# Base64 Encoding
b64_header=$(pack "$header" | base64url_encode)
b64_payload=$(pack "$payload" | base64url_encode)
signature=$(echo -n $b64_header.$b64_payload | openssl dgst -sha256 -sign private-key.pem | base64url_encode)
# Export JWT
echo $b64_header.$b64_payload.$signature > jwt.txt
# Create JWK from public key
if [ ! -d pem-jwk ]; then
  # A tool to convert PEM to JWK
  npm install pem-jwk
fi
jwk=$(pem-jwk ~/.ssh/moneyhub_pub.pem)
# Add additional fields
jwk=$(echo '{"use":"sig"}' $jwk $header | jq -cs add)
# Export JWK
echo '{"keys":['$jwk']}'| jq . > jwks.json

echo "--- JWT ---"
cat jwt.txt
echo -e "\n--- JWK ---"
jq . jwks.json