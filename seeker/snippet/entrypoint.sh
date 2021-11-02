#date: 2021-11-02T16:57:25Z
#url: https://api.github.com/gists/40d8fb628d726f267cdc840b7fbddb6e
#owner: https://api.github.com/users/TimyStream

#! /bin/sh
FILE="$proxyVersion.jar"
if [ -f "$FILE"]; then
    echo "Starting Proxy with Version: $proxyVersion"
    echo ""
    java "$minRam $maxRam $otherVariables -jar $proxyVersion.jar"
fi