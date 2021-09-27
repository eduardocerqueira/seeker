#date: 2021-09-27T17:07:29Z
#url: https://api.github.com/gists/829b1e03fa794ca6beb16dfe0155508c
#owner: https://api.github.com/users/scottj

curl -sx 127.0.0.1:8080 http://burp/cert \
| openssl x509 -inform DER \
| sudo tee /usr/local/share/certificates/burp.crt \
&& sudo update-ca-certificates