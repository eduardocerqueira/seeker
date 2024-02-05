#date: 2024-02-05T16:56:06Z
#url: https://api.github.com/gists/a970bd012a7fec1d3c1fe8ce51af3acf
#owner: https://api.github.com/users/jobrunner

#!/bin/sh

echo "Import certificate from file '${1}' into osx system keychain with full trust"
subject_line=$(openssl x509 -noout -subject -in "${1}")
common_name=$(echo "${subject_line}" | sed -En "s/subject=CN = (.*)/\1/p")
echo "Common Name of the certificate: $common_name"
sudo security add-certificate -k "/Library/Keychains/System.keychain" "${1}"
sha1=$(sudo security find-certificate -c "${common_name}" -a -Z | sed -En "s/SHA-1 hash: (.*)/\1/p")
sudo security add-trusted-cert -d \
	-r trustRoot \
	-k "/Library/Keychains/System.keychain" \
	-p ssl \
	-p basic \
	-p codeSign \
	-p pkgSign \
	-p eap \
	-p smime \
	-s $sha1 \
  "${1}"
