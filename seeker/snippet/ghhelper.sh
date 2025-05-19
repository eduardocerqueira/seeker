#date: 2025-05-19T17:02:24Z
#url: https://api.github.com/gists/66a6cac6daa84bdcc50fe0a89af71999
#owner: https://api.github.com/users/jookovjook

#!/bin/bash

# https://gist.githubusercontent.com/jookovjook/66a6cac6daa84bdcc50fe0a89af71999/raw/ghhelper.sh
# wget -O ghhelper.sh https://gist.githubusercontent.com/jookovjook/66a6cac6daa84bdcc50fe0a89af71999/raw/ghhelper.sh && chmod +x ghhelper.sh && ./ghhelper.sh

if [ -n "$GH_TOKEN" ]; then
    echo "Using GH_TOKEN to set git global config"
    git config --global url."https: "**********"://github.com/"
else
    echo "GH_TOKEN not defined, not setting git global config"
fietting git global config"
fi