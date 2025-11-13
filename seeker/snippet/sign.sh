#date: 2025-11-13T16:59:57Z
#url: https://api.github.com/gists/fcd988d0253c51104e8bbe050e057891
#owner: https://api.github.com/users/mykulyak

#!/usr/bin/env bash

CERTIFICATE_NAME="...."
/usr/bin/codesign --sign "$CERTIFICATE_NAME" --timestamp --options runtime --verbose /path/to/your/cli