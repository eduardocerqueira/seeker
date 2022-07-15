#date: 2022-07-15T17:06:09Z
#url: https://api.github.com/gists/0539369d846a22a1b70939388f4534fc
#owner: https://api.github.com/users/dode5656

#!/bin/bash

# Use ghcr.io/dode5656/yolks:java_17 as custom docker image

#Initialise startServer function
startServer () {

    java -Xms2G -Xmx2G -XX:+UseG1GC -XX:+ParallelRefProcEnabled -XX:MaxGCPauseMillis=200 -XX:+UnlockExperimentalVMOptions -XX:+DisableExplicitGC -XX:+AlwaysPreTouch -XX:G1NewSizePercent=30 -XX:G1MaxNewSizePercent=40 -XX:G1HeapRegionSize=8M -XX:G1ReservePercent=20 -XX:G1HeapWastePercent=5 -XX:G1MixedGCCountTarget=4 -XX:InitiatingHeapOccupancyPercent=15 -XX:G1MixedGCLiveThresholdPercent=90 -XX:G1RSetUpdatingPauseTimePercent=5 -XX:SurvivorRatio=32 -XX:+PerfDisableSharedMem -XX:MaxTenuringThreshold=1 -Dusing.aikars.flags=https://mcflags.emc.gs -Daikars.new.flags=true -jar $SERVER_JARFILE --nogui

}

#Check if versionforscript file exists, if it does then get the variables from it
if [ -e versionforscript ]; then
    PREV_VERSION=$(sed -e 's/^"//' -e 's/"$//' <<< $(awk -F= '$1=="VERSION"{print $2;exit}' versionforscript))
    PREV_BUILD=$(sed -e 's/^"//' -e 's/"$//' <<< $(awk -F= '$1=="BUILD"{print $2;exit}' versionforscript))
fi

#Check if their server version exists, if it doesnt default to latest
VER_EXISTS=$(curl -s https://api.papermc.io/v2/projects/paper | tac | tac | jq -r --arg VERSION "$MINECRAFT_VERSION" '.versions[] | contains($VERSION)' | grep -m1 true)
if [ "$VER_EXISTS" == "true" ]; then
    VERSION=$MINECRAFT_VERSION;
else 
    #Default to latest version
    VERSION=$(sed -e 's/^"//' -e 's/"$//' <<< $(curl -s https://api.papermc.io/v2/projects/paper | tac | tac | jq ".versions[-1]"))
fi

#Grab latest build
LATEST_BUILD=$(sed -e 's/^"//' -e 's/"$//' <<< $(curl -s "https://api.papermc.io/v2/projects/paper/versions/$VERSION/builds" | tac | tac | jq -r '.builds[-1].build'))
if [ "$LATEST_BUILD" == "$PREV_BUILD" ] && [ "$VERSION" == "$PREV_VERSION" ] && [ ! -z ${PREV_BUILD+x} ] && [ ! -z ${PREV_VERSION+x} ]; then
    startServer
    exit 0
fi

#Compile a jar name for the download
JAR_NAME="paper-$VERSION-$LATEST_BUILD.jar"

#Save version information for next script run
printf "VERSION=$VERSION\nBUILD=$LATEST_BUILD" > versionforscript

# Move old server jarfile to server.jar.old
mv -f $SERVER_JARFILE $SERVER_JARFILE.old

#Download the server jarfile from Paper
curl -o $SERVER_JARFILE "https://api.papermc.io/v2/projects/paper/versions/$VERSION/builds/$LATEST_BUILD/downloads/$JAR_NAME"
echo "Downloaded latest build ($LATEST_BUILD), version $VERSION"
startServer
exit 0