#date: 2025-11-10T16:54:31Z
#url: https://api.github.com/gists/096f76003e8d3540dda7961198c19e7e
#owner: https://api.github.com/users/joakime

#/usr/bin/env bash

# See https://api.adoptium.net/q/swagger-ui/ for documentation about api.adoptium.net

# For valid list of values for the following, see documentation about API

# See: https://api.adoptium.net/v3/info/available_releases
JVM_FEATURE="$1"

if [ -z $JVM_FEATURE ] ; then
    echo "ERROR  : No JVM feature provided."
    echo "Usage  : ./getjvm.sh [jvm-feature]"
    echo "Example: ./getjvm.sh 17" 
    echo ""
    echo "Here's the list of the currently available jvms..."
    curl https://api.adoptium.net/v3/info/available_releases
    exit -1
fi

JVM_RELEASE_TYPE="ga"
JVM_OS="linux"
JVM_ARCH="x64"
JVM_TYPE="jdk"
JVM_IMPL="hotspot"
JVM_HEAP_SIZE="normal"
JVM_VENDOR="eclipse"


curl --location \
     --remote-name \
     --remote-header-name \
     https://api.adoptium.net/v3/binary/latest/${JVM_FEATURE}/${JVM_RELEASE_TYPE}/${JVM_OS}/${JVM_ARCH}/${JVM_TYPE}/${JVM_IMPL}/${JVM_HEAP_SIZE}/${JVM_VENDOR}

