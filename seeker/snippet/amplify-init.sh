#date: 2022-04-15T17:07:14Z
#url: https://api.github.com/gists/a8cd6071d250693fdae73c84bc5b23f7
#owner: https://api.github.com/users/josefaidt

#!/bin/bash
set -e
IFS='|'

AMPLIFY_APP_ID='d3mevdutihu9jw'
AMPLIFY_ENVIRONMENT='dev'

AWSCLOUDFORMATIONCONFIG="{\
\"configLevel\":\"project\",\
\"useProfile\":true,\
\"profileName\":\"default\",\
\"region\":\"us-east-1\"\
}"
AMPLIFY="{\
\"appId\":\"$AMPLIFY_APP_ID\",\
\"envName\":\"$AMPLIFY_ENVIRONMENT\",\
}"

PROVIDERS="{\
\"awscloudformation\":$AWSCLOUDFORMATIONCONFIG}"

amplify init \
  --amplify $AMPLIFY \
  --providers $PROVIDERS \
  --yes || true
