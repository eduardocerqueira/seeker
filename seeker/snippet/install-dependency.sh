#date: 2022-07-13T17:17:32Z
#url: https://api.github.com/gists/09ca2c4b366ac46652b9962b550537d8
#owner: https://api.github.com/users/gdcmarinho

if [ -z "$1" ]; then
    echo "groupId not specified"
    echo "Finishing..."
    exit 1
elif [ -z "$2" ]; then
    echo "artifactId not specified"
    echo "Finishing..."
    exit 1
elif [ -z "$3" ]; then
    echo "Lib version not specified"
    echo "Finishing..."
    exit 1
fi

echo "Installing dependency: $1:$2:$3"

xmlstarlet ed -L -N p="http://maven.apache.org/POM/4.0.0" -s /p:project/p:dependencies -t elem -n dependency -v "" \
    -s //dependency -t elem -n "groupId" -v "$1" \
    -s //dependency -t elem -n "artifactId" -v "$2" \
    -s //dependency -t elem -n "version" -v "$3" \
    pom.xml