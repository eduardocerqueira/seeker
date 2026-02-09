#date: 2026-02-09T17:44:24Z
#url: https://api.github.com/gists/bf2b57545c4f041447ffa7be436f6f7b
#owner: https://api.github.com/users/jspeer

#!/bin/bash

# Place the contents of this file in /config/scripts/firstboot.d/acme.sh and edit the settings below.
# Run the script and your router should have a valid ACME SSL certificate.

#### SETTINGS NEED TO BE CORRECT FOR A VALID CERTIFICATE TO BE ISSUED
#### THESE SETTINGS HAVE BEEN TESTED AND WORKING ON A step-ca ACME SERVER

#### ACME Server BASE URL
ACME_CA_DOMAIN="https://ca.mydomain.com"

#### ACME Server Directory URL
ACME_CA_SERVER_DIRECTORY="${ACME_CA_DOMAIN}/acme/acme/directory"

#### Root CA certificate filename
ACME_CA_ROOT_BUNDLE="roots.pem"

#### Root CA certificate file URL
ACME_CA_ROOT_BUNDLE_URL="${ACME_CA_DOMAIN}/${ACME_CA_ROOT_BUNDLE}"

#### Duration in days the CA will issue the certificate for.
#### Please note, this will not request a certificate for this duration, this is how long frequently you want acme.sh to request a new certificate.
#### Check your ACME server configuration to know exactly how long your default certificates are issued for.
ACME_CA_CERT_DURATION_IN_DAYS=1

#### The hostname of your EdgeOS router
SERVER_HOSTNAME="router.mydomain.com"
SERVER_CERTIFICATE="server.pem"

#### END OF SETTINGS

ACME_HOME="/config/.acme.sh"

#### Bail out if we're already got ACME set up and running
if [ -f "/config/ssl/${SERVER_CERTIFICATE}" ]; then
        echo "Certificate already installed to /config/ssl/${SERVER_CERTIFICATE}. If this is in error, delete the certificate and re-run $0."
        echo "Note: If you also want to reset ACME state, delete ${ACME_HOME} after removing the certificate."
        exit 0
fi

# Install ACME.SH from official source if not already installed
# This will also establish the cron job to ensure the certificate is renewed
if [ ! -d "$ACME_HOME" ]; then
        mkdir ${ACME_HOME}
        /usr/bin/curl https://get.acme.sh | /bin/sh -s -- home ${ACME_HOME}
fi

# Get root CA cert
if [ ! -f "/config/ssl/${ACME_CA_ROOT_BUNDLE}" ]; then
        /usr/bin/curl -k ${ACME_CA_ROOT_BUNDLE_URL} -o /config/ssl/${ACME_CA_ROOT_BUNDLE}
fi

# Issue first cert
${ACME_HOME}/acme.sh --issue --home ${ACME_HOME} -d ${SERVER_HOSTNAME} --standalone --server ${ACME_CA_SERVER_DIRECTORY} --ca-bundle /config/ssl/${ACME_CA_ROOT_BUNDLE} --days ${ACME_CA_CERT_DURATION_IN_DAYS} --pre-hook "systemctl stop lighttpd.service" --post-hook "cat /config/ssl/cert.* > /config/ssl/${SERVER_CERTIFICATE}; systemctl start lighttpd.service" --key-file /config/ssl/cert.key --fullchain-file /config/ssl/cert.p

# Update EdgeOS config
source /opt/vyatta/etc/functions/script-template

# Enter configuration mode and set changes
configure
delete service gui ca-file
delete service gui cert-file
set service gui ca-file /config/ssl/${ACME_CA_ROOT_BUNDLE}
set service gui cert-file /config/ssl/${SERVER_CERTIFICATE}

# Commit and save changes
commit
save
exit
