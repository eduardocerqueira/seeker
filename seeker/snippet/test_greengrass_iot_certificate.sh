#date: 2023-01-31T17:06:16Z
#url: https://api.github.com/gists/19f141aae3094c233ff3ecfdbe1795e7
#owner: https://api.github.com/users/bml1g12

#!/usr/bin/env bash
set -Eeuxo pipefail
# If running this script on a Greengrass v2 device, it should return a dictionary containing temporary credentials.
# If it fails, check if the THING_NAME, ROLE_ALIAS and IOT_GET_CREDENTIAL_ENDPOINT are correct
# IOT_GET_CREDENTIAL_ENDPOINT can be verified via running ` aws iot describe-endpoint --endpoint-type iot:CredentialProvider --output text`
# on a different device with sufficent IAM to run this command, in the same AWS account and region. 

THING_NAME=`sudo cat /greengrass/v2/config/effectiveConfig.yaml | grep -i thingName | awk '{ print $2 }' | tr -d '"'`
ROLE_ALIAS=`sudo cat /greengrass/v2/config/effectiveConfig.yaml | grep -i rolealias | awk '{ print $2 }' | tr -d '"'`
IOT_GET_CREDENTIAL_ENDPOINT=`sudo cat /greengrass/v2/config/effectiveConfig.yaml | grep iotCredEndpoint | awk '{print $2}' | tr -d '"'`

PRIVATE_KEY_PATH=/greengrass/v2/privKey.key
CA_CERT_PATH=/greengrass/v2/rootCA.pem
CERT_PATH=/greengrass/v2/thingCert.crt

curl --cert ${CERT_PATH} --key ${PRIVATE_KEY_PATH} -H "x-amzn-iot-thingname: ${THING_NAME}" --cacert ${CA_CERT_PATH} https://${IOT_GET_CREDENTIAL_ENDPOINT}/role-aliases/${ROLE_ALIAS}/credentials
