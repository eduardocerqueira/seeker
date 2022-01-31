#date: 2022-01-31T17:04:43Z
#url: https://api.github.com/gists/c503b21377d929adf2214608b1900eaf
#owner: https://api.github.com/users/tothi

#!/bin/bash
#
# openvpn2 wrapper for supporting Dynamic Challenge (in openvpn 2.x)
#

# Dynamic Challenge:
#   CRV1:<FLAGS>:<STATE_ID>:<BASE64_USERNAME>:<CHALLENGE_TEXT>
# Dynamic Challenge response:
#   Username: [username decoded from challenge, probably equals to the original username]
#   Password: CRV1::<STATE_ID>::<RESPONSE_TEXT>

# STAGE1: Connect with auth-user-pass
#   --> AUTH_FAILED with STATE_ID and Username + send OTP code
# STAGE2: Connect with auth-user-pass
#    Username: Username recevied from STAGE1
#    Password: CRV1::<STATE_ID received from STAGE1>::<OTP RESPONSE_TEXT received>
#  --> AUTH SUCCESS

if [ $# -ne 1 ]; then
  echo "Usage: $0 <ovpn config file>"
  exit 1
fi

OPENVPN_CONFIG=$1
logfile=$(mktemp /tmp/openvpn2-wrapper.XXXXXX)

echo "[*] STAGE1: Connecting and authenticating with password for initiating Dynamic Challenge..."

openvpn --config $OPENVPN_CONFIG --verb 3 --log $logfile

if ( cat $logfile | grep "AUTH_FAILED,CRV1:" >/dev/null ); then
  echo "[+] Dynamic Challenge initiated..."
else
  echo "[!] no Dynamic Challenge :("
  exit 1
fi

STATE_ID=`sed -ne 's/^.*: AUTH_FAILED,//p' $logfile | cut -d: -f3`
USERNAME=`sed -ne 's/^.*: AUTH_FAILED,//p' $logfile | cut -d: -f4 | base64 -d`

echo "[+] Extracted STATE_ID=${STATE_ID} and Username=${USERNAME}"
echo

read -p "Enter Dynamic Challenge OTP passcode: " OTP
echo

rm -f $logfile
AUTHFILE=$(mktemp /tmp/openvpn2-wrapper.XXXXXX)

cat <<EOF > $AUTHFILE
$USERNAME
CRV1::${STATE_ID}::${OTP}
EOF

echo "[+] STAGE2: Temp file with stage2 (OTP) password created"
echo "[*] STAGE2: Cleaning up temp file after 15 seconds..."
( sleep 15; rm -f $AUTHFILE; echo "[+] STAGE2: Cleaned up temp files" ) &

echo "[*] STAGE2: Connecting and authenticating with OTP pass..."
openvpn --config $OPENVPN_CONFIG --auth-user-pass $AUTHFILE

exit 0
