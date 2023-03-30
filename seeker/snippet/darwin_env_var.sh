#date: 2023-03-30T17:07:15Z
#url: https://api.github.com/gists/df2206f61e2b2ee9529452b331086650
#owner: https://api.github.com/users/n8felton

#!/bin/sh

# User path
/usr/bin/getconf DARWIN_USER_DIR

# Temp directory
# Note - $TMPDIR is the same as this
/usr/bin/getconf DARWIN_USER_TEMP_DIR

# Cache directory
/usr/bin/getconf DARWIN_USER_CACHE_DIR