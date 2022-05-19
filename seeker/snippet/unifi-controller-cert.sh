#date: 2022-05-19T17:30:02Z
#url: https://api.github.com/gists/468480fc7751be8206f0332d9fa44303
#owner: https://api.github.com/users/mislav

cert="mycert.pem"
cert_key="mycert.key"
root_ca="root.pem"
pfx_password="whatever" # this doesn't really matter

# Generate a PKCS#12 store of certificate, private key, and root certificate
openssl pkcs12 -export \
  -in "$cert" -inkey "$cert_key" -CAfile "$root_ca" -caname root \
  -out unifi.pfx -passout pass:"$pfx_password" \
  -name unifi

# This converts a PKCS12 store to Java KeyStore file named "keystore" with password "aircontrolenterprise"
keytool -importkeystore \
  -srckeystore unifi.pfx -srcstoretype PKCS12 -srcstorepass "$pfx_password" \
  -deststorepass aircontrolenterprise -destkeypass aircontrolenterprise -destkeystore keystore \
  -alias unifi

# Note: "Keytool" is a Java utility and might not be immediately available on your OS. However, it's present in
# the Unifi Controller docker container, and I was able to access it by opening a shell in the container:
# > docker-compose run unifi-controller /bin/bash

# Now move `keystore` to an appropriate location. Within the `lscr.io/linuxserver/unifi-controller:latest` container,
# that location is `/config/data/keystore`. (It's fine to overwrite the old keystore.)