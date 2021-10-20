#date: 2021-10-20T17:02:07Z
#url: https://api.github.com/gists/f252cec4357c14f8a8860eb3bee02623
#owner: https://api.github.com/users/orinocoz

#!/bin/sh -e

KERIO_SSLDIR="/opt/kerio/mailserver/sslcert"

TOPDIR="/etc/LetsEncrypt"
TSTAMP="$(date +%Y-%m-%d-%H%M)"

NEWDIR="$TOPDIR/$TSTAMP"
CURDIR="$TOPDIR/current"

Update_current() { cd "$TOPDIR/" && ln -nfs "$TSTAMP" current; }

mkdir -p "$NEWDIR"

test -e "$CURDIR" && cmd="renew" || cmd="issue"

/home/acme.sh/acme.sh \
    --$cmd \
    --force \
    --domain mail.remotesrv.ru \
    --webroot /var/www/letsencrypt/ \
    --log             "$NEWDIR/acme.log" \
    --cert-file       "$NEWDIR/cert.pem" \
    --key-file        "$NEWDIR/key.pem"  \
    --ca-file         "$NEWDIR/ca.pem"   \
    --fullchain-file  "$NEWDIR/fullchain.pem" \
    --reloadcmd "touch $NEWDIR/success.flag"

test -s "$NEWDIR/success.flag" || exit 1

Update_current

cp -p "$NEWDIR/fullchain.pem" "$KERIO_SSLDIR/LetsEncrypt.crt"
cp -p "$NEWDIR/key.pem"       "$KERIO_SSLDIR/LetsEncrypt.key"

systemctl reload  nginx
systemctl restart kerio-connect.service

## END ##