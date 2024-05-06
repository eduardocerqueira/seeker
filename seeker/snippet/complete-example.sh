#date: 2024-05-06T17:11:25Z
#url: https://api.github.com/gists/28154474441720e45e7e296375fc3383
#owner: https://api.github.com/users/burgil

#!/bin/bash

OWNER="burgil"
REPO_NAME="Zeteor"

get_latest_commit() {
    curl -s "https://api.github.com/repos/$OWNER/$REPO_NAME/commits/main" | grep '"sha"' | head -n 1 | cut -d '"' -f 4
}
LAST_COMMIT=$(get_latest_commit)
while true; do
    LATEST_COMMIT=$(get_latest_commit)
    echo "LATEST COMMIT: $LATEST_COMMIT"
    if [ "$LATEST_COMMIT" != "$LAST_COMMIT" ]; then
        echo "Changes detected in the repository!"
        LAST_COMMIT=$LATEST_COMMIT
        echo "Restarting Website..."
        # ps aux | grep "ZeteorMainWebsite" | grep -v grep | awk '{print $2}' | xargs -I{} kill -9 {}
        # cd ..
        echo "Fetching Updates..."
        # git fetch
        echo "Pulling Updates..."
        # git pull
        # cd website
        echo "Starting Website..."
        # nohup npm start &
        echo "Website Restarted!"
    fi
    sleep 30
done