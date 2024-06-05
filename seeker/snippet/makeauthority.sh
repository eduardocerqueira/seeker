#date: 2024-06-05T17:03:06Z
#url: https://api.github.com/gists/df8e37a1a34f69088e8c2f3aba16a50a
#owner: https://api.github.com/users/Arthur1337

# Run this once
openssl genrsa -des3 -out ca.key 4096
openssl req -new -x509 -days 365 -key ca.key -out ca.crt