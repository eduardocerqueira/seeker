#date: 2024-12-24T16:41:01Z
#url: https://api.github.com/gists/c54a8f6ae2831b8d7b2cc926c6c936aa
#owner: https://api.github.com/users/danilogco

#!/bin/bash

APP_FOLDER=/opt
TMP_FOLDER=/tmp

rm -rf $APP_FOLDER/dbeaver*
wget "https://dbeaver.io/files/dbeaver-ce-latest-linux.gtk.x86_64-nojdk.tar.gz" -P $TMP_FOLDER/
tar -xvzf $TMP_FOLDER/dbeaver-ce-latest-linux.gtk.x86_64-nojdk.tar.gz -C $APP_FOLDER/
rm -rf $TMP_FOLDER/dbeaver-ce-latest-linux.gtk.x86_64-nojdk.tar.gz