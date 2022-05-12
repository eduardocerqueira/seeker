#date: 2022-05-12T17:03:20Z
#url: https://api.github.com/gists/db60030638b795b9de47fbdb3c076b2e
#owner: https://api.github.com/users/seesemichaelj

#!/bin/bash

# only used in windows for cross compiling
export LINUX_MULTIARCH_ROOT=/c/UnrealToolchains/v19_clang-11.0.1-centos7/
if [ "$(uname)" != "Darwin" ]; then
  ${LINUX_MULTIARCH_ROOT}x86_64-unknown-linux-gnu/bin/clang++ -v
fi

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_FILE=$(/usr/bin/find ${SCRIPT_DIR} -maxdepth 1 -name *.uproject)
PROJECT_NAME=$(basename ${PROJECT_FILE} .uproject)

ENGINE_VERSION=UE5_Source
TARGET=$1 # "client" or "server"
CONFIG=$2 # "Debug", "DebugGame", "Development", "Test", or "Shipping"
OS_OPTION=$3 # only "linux" for cross compile
ARCHIVE_DIR=${SCRIPT_DIR}\\builds\\${CONFIG}\\${TARGET}

AS_BASE_DIR=Script

# Parameters for SFTPing builds to another machine
SSH_USER="<INSERT SSH USER>"
SSH_HOST="<INSERT SSH HOST>"
SSH_ARCHIVE_FOLDER="<INSERT SSH FOLDER>"

if [ "$(uname)" == "Darwin" ]; then
  UAT_PATH="/Users/${USERNAME}/work/${ENGINE_VERSION}/Engine/Build/BatchFiles/RunUAT.sh"
  TARGET_PLATFORM=Mac
  PLATFORM_NAME=Mac
  EXE_EXT=".app"
  SSH_KEY="/Users/${USERNAME}/.ssh/id_ed25519_jenkins"
  7ZIP_PATH="/usr/local/bin/7zz"
else
  UAT_PATH="/d/epic/engine/${ENGINE_VERSION}/Engine/Build/BatchFiles/RunUAT.bat"

  # windows handles linux cross compile
  if [ "${OS_OPTION}" == "linux" ]; then
    TARGET_PLATFORM=Linux
    PLATFORM_NAME=Linux
    EXE_EXT=".sh"
  else
    TARGET_PLATFORM=Win64
    PLATFORM_NAME=Windows
    EXE_EXT=".exe"
  fi
  SSH_KEY="/c/Users/${USERNAME}/.ssh/id_ed25519_jenkins"
  7ZIP_PATH="/c/Program\ Files/7-Zip/7z"
fi

if [ "${TARGET}" == "server" ]; then
  PLATFORM_PARAM="-serverplatform=${TARGET_PLATFORM}"
  CLIENT_PARAM="-noclient"
  SERVER_PARAM="-server"
  TARGET_NAME="Server"
else
  PLATFORM_PARAM="-platform=${TARGET_PLATFORM}"
  CLIENT_PARAM="-client"
  SERVER_PARAM=""
  TARGET_NAME="Client"
fi

BUILD_SUBDIR=${PLATFORM_NAME}${TARGET_NAME}
OUT_NAME=${PROJECT_NAME}${TARGET_NAME}
OUT_EXE=${OUT_NAME}${EXE_EXT}
BUILD_DIR=${ARCHIVE_DIR}/${BUILD_SUBDIR}

ExecuteBuild() {
  ARGS="\
    BuildCookRun \
    -project=${PROJECT_FILE} \
    -noP4 \
    ${PLATFORM_PARAM} \
    ${CLIENT_PARAM} \
    ${SERVER_PARAM} \
    -${TARGET}config=${CONFIG} \
    -cook \
    -build \
    -stage \
    -pak \
    -iostore \
    -cook4iostore \
    -archive \
    -archivedirectory=${ARCHIVE_DIR} \
    -separatedebuginfo \
    -unattended"

  "${UAT_PATH}" ${ARGS}
  return $?
}

ExecuteBuild
buildStatus=$?
if [ "${buildStatus}" != "0" ]; then
  exit ${buildStatus}
fi

mkdir -p ${BUILD_DIR}/${PROJECT_NAME}/${AS_BASE_DIR}

# copy the cache for the AS binds
cp ${AS_BASE_DIR}/Binds.Cache* ${BUILD_DIR}/${PROJECT_NAME}/${AS_BASE_DIR}

# find all the unique dirs in AS_BASE_DIR that have AS code
AS_DIRS=()
while IFS=  read -r -d $'\0'; do
  NEXTDIR=$(echo "${REPLY}" | awk 'BEGIN { FS = "/" } ; { print $2 }')
  AS_DIRS+=("$NEXTDIR")
done < <(/usr/bin/find ${AS_BASE_DIR} -name *.as -print0)
AS_DIRS=($(for dir in "${AS_DIRS[@]}"; do echo "${dir}"; done | /usr/bin/sort -u))

# copy AS code to the build dir
for dir in "${AS_DIRS[@]}"; do
  cp -r ${AS_BASE_DIR}/${dir} ${BUILD_DIR}/${PROJECT_NAME}/${AS_BASE_DIR}
done

pushd ${BUILD_DIR}

if [ "${OS_OPTION}" == "linux" ]; then
  /c/Windows/System32/bash -c "./${OUT_EXE} -nullrhi -as-generate-precompiled-data || echo Finished"
  rm ./${PROJECT_NAME}/Binaries/Linux/core || echo "Linux core file was not detected, so it wasn't deleted"
else
  ./${OUT_EXE} -nullrhi -as-generate-precompiled-data || echo Finished
fi

# remove AS code to the build dir
for dir in "${AS_DIRS[@]}"; do
  rm -rf ./${PROJECT_NAME}/${AS_BASE_DIR}/${dir}
done

# remove JITTED code
rm -rf AS_JITTED_CODE

# remove Saved directory
rm -rf ./Engine/Saved
rm -rf ./${PROJECT_NAME}/Saved

cd ..

zipName=${PLASTICSCM_CHANGESET_ID}_${PLATFORM_NAME}_${OUT_NAME}.zip
${7ZIP_PATH} a -r -tzip ${zipName} ${BUILD_SUBDIR}

sftp -i ${SSH_KEY} ${SSH_USER}@${SSH_HOST} <<END
cd ${SSH_ARCHIVE_FOLDER}
put ${zipName}
END

popd
