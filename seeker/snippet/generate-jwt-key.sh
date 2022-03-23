#date: 2022-03-23T17:05:27Z
#url: https://api.github.com/gists/7321dc6d2064a3ccd8b97037012d1f9c
#owner: https://api.github.com/users/paveltretyakovru

ssh-keygen -t rsa -b 4096 -m PEM -f ./certs/jwt.key
# Don't use codephase
openssl rsa -in jwtRS256.key -pubout -outform PEM -out ./certs/jwt.pub
