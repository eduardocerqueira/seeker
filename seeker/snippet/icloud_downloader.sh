#date: 2023-03-24T16:55:49Z
#url: https://api.github.com/gists/d0d4e6b7d2570409bc7d6ec92350a086
#owner: https://api.github.com/users/adamzachyang

#!/bin/bash
# for an icloud file with link "https://www.icloud.com/iclouddrive/<ID>#<Filename>
# requires jq installed, install with 'brew install jq'
# put <ID> below to download and chmod u+x the script to make executable.

ID = " "
URL=$(curl 'https://ckdatabasews.icloud.com/database/1/com.apple.cloudkit/production/public/records/resolve' \
  --data-raw '{"shortGUIDs":[{"value":"$ID"}]}' --compressed | jq -r '.results[0].rootRecord.fields.fileContent.value.downloadURL')
Echo $url
curl "$URL" -o myfile.ext