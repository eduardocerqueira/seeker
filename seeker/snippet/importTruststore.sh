#date: 2021-08-31T13:15:44Z
#url: https://api.github.com/gists/a1723fd6cdb221f4173fe45f2c7ad2e8
#owner: https://api.github.com/users/jaystile

#!/bin/bash
# Template for importing one truststore into another which transfers all aliases and certs.
src_keystore=./srctruststore.jks
src_store_password='src_store_password'
dest_keystore=./dest_keystore.jks
dest_store_password='dest_store_password'
keytool -importkeystore -v -noprompt \
-srckeystore ${src_keystore} -srcstorepass ${src_store_password} \
-destkeystore ${dest_keystore} -deststorepass ${dest_store_password}  
