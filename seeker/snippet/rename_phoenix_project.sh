#date: 2021-12-22T16:58:46Z
#url: https://api.github.com/gists/7aa0bf9da3f48fe6e2af42427963e22b
#owner: https://api.github.com/users/oliverandrich

#!/bin/bash

set -e

CURRENT_NAME="Zauberantrag"
CURRENT_OTP="zauberantrag"

NEW_NAME="Wunderantrag"
NEW_OTP="wunderantrag"

ack -l $CURRENT_NAME --ignore-file=is:rename_phoenix_project.sh | xargs sed -i '' -e "s/$CURRENT_NAME/$NEW_NAME/g"
ack -l $CURRENT_OTP --ignore-file=is:rename_phoenix_project.sh | xargs sed -i '' -e "s/$CURRENT_OTP/$NEW_OTP/g"

git mv lib/$CURRENT_OTP lib/$NEW_OTP
git mv lib/$CURRENT_OTP.ex lib/$NEW_OTP.ex
git mv lib/${CURRENT_OTP}_web lib/${NEW_OTP}_web
git mv lib/${CURRENT_OTP}_web.ex lib/${NEW_OTP}_web.ex
git mv test/$CURRENT_OTP test/$NEW_OTP
git mv test/${CURRENT_OTP}_web test/${NEW_OTP}_web
