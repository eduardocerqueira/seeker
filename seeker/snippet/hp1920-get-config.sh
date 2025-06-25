#date: 2025-06-25T17:14:40Z
#url: https://api.github.com/gists/a20bb25349b37353fae32d4c944ff410
#owner: https://api.github.com/users/fvfl

#!/bin/bash
#
# Simple script to download the running configuration from the HP 1920S switch
# through the HTTP "APIs"
#
# Run it as:
#  $ ./hp1920-getconfig.sh --host="10.1.2.3" --user="admin" --pass="hello" --file=startup-config
#
# Attila Sukosd <attila@airtame.com>
#
# HB 19.05.2020
# adapted for it to work with J9979A (HPE OC Switch 1820 8G)

HOST=""
USER="admin"
PASS=""

for i in "$@"; do
  case $i in
    -h=*|--host=*)
    HOST="${i#*=}"
    shift
    ;;
    -u=*|--user=*)
    USER="${i#*=}"
    shift
    ;;
    -p= "**********"=*)
    PASS="${i#*=}"
    shift
    ;;
    -f=*|--file=*)
    FILE="${i#*=}"
    shift
    ;;
    *)
          # unknown option
    ;;
  esac;
done;

if [ "$HOST" == "" -o "$FILE" == "" ]; then
	echo "Error. You need to specify at least the host with --host and the output file with --file";
	exit 1;
fi;

echo -n "Logging in to the HP switch... "
# Login
CS=$(curl -v -d "username=${USER}&password=$PASS" http: "**********"

if [ "$?" -ne 0 ]; then
  echo "Error."
  exit 1;
fi;
echo "OK"

# Format the cookies correctly
H=$(echo $CS |sed 's/ /; /g')

TS=$(date +%s000)

echo -n "Requesting to download config... "
# Request config download
PARAMS=$(curl -d "file_type_sel[]=config&http_token=$TS" -H "Referer: "**********"://$HOST/htdocs/pages/base/file_upload_modal.lsp?help=/htdocs/lang/en_us/help/base/help_file_transfer.lsp&filetypes=6&protocol=6" -H "Cookie: $H" "**********"://$HOST/htdocs/lua/ajax/file_upload_ajax.lua?protocol=6 2>/dev/null)

if [ "$(echo $PARAMS |grep '"successful": "ready",')" == "" ]; then
  echo "Error."
  echo $PARAMS
  exit 1
fi
echo "OK"

PARAMS2=$(echo $PARAMS | cut -d '?' -f 2 | cut -d '"' -f 1)

echo -n "Downloading config... "
curl -H "Referer: http://$HOST/htdocs/pages/base/file_upload_modal.lsp?help=/htdocs/lang/en_us/help/base/help_file_transfer.lsp&filetypes=6&protocol=6" -H "Cookie: $H" "http://$HOST/htdocs/pages/base/file_http_download.lsp?$PARAMS2" -o "$FILE" 2>/dev/null

if [ "$?" -ne 0 ]; then
	echo "Error."
	exit 1
fi
# needed in order to finish download process properly
curl -H "Referer: http://$HOST/htdocs/pages/main/main.lsp" -H "Cookie: $H" "http://$HOST/htdocs/pages/base/file_http_download.lsp?${PARAMS2}&remove=true" 2>/dev/null

echo "OK. Saved to $FILE."
# logoff
curl  -H "Referer: http://$HOST/htdocs/pages/main/main.lsp" -H "Cookie: $H" http://$HOST/htdocs/pages/main/logout.lsp 2>/dev/null | sed -e 's/<.*>//'
mv "$FILE" "${FILE}-${HOST}"
/dev/null | sed -e 's/<.*>//'
mv "$FILE" "${FILE}-${HOST}"
