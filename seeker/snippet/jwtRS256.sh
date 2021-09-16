#date: 2021-09-16T17:05:28Z
#url: https://api.github.com/gists/8ddc043b9ee327861f4dd45fcd8f5f20
#owner: https://api.github.com/users/tapsu01

ssh-keygen -t rsa -b 4096 -m PEM -f jwtRS256.key
# Don't add passphrase
openssl rsa -in jwtRS256.key -pubout -outform PEM -out jwtRS256.key.pub
cat jwtRS256.key
cat jwtRS256.key.pub