#date: 2023-06-27T16:40:55Z
#url: https://api.github.com/gists/5e38528ec62fdc5925293d97716f7102
#owner: https://api.github.com/users/ragevna

#!/bin/bash
find /etc/kubernetes/pki/ -type f -name "*.crt" -print|egrep -v 'ca.crt$'|xargs -L 1 -t  -i bash -c 'openssl x509  -noout -text -in {}|grep After'