#date: 2022-06-14T17:10:29Z
#url: https://api.github.com/gists/b0fc4a6055ffd2263a465ab2677ab799
#owner: https://api.github.com/users/maykefreitas

#!/bin/bash
set -e

if [ -x "$(command -v yum)" ]; then
  export LINUX_DISTRO='centos'
fi
if [ -x "$(command -v apt-get)" ]; then
  export LINUX_DISTRO='debian'
fi
if [ -x "$(command -v apk)" ]; then
  export LINUX_DISTRO='alpine'
fi

echo LINUX_DISTRO: ${LINUX_DISTRO}

if [ ! -z $1 ]; then
  PREFIX="$1-"
else
  PREFIX=""
fi

export AWS_ACCESS_KEY_ID=
export AWS_SECRET_ACCESS_KEY=
export AWS_DEFAULT_REGION=
export AWS_S3_BUCKET_NAME=
export YARN_CACHE_DIR_LOCATION=$(yarn cache dir)
export NODE_MODULES_GZ_FILE_PREFIX="${PREFIX}node-modules-cache"
export NODE_MODULES_CACHE_HASH=$(sha256sum yarn.lock | grep -o '^\S\+')
export NODE_MODULES_GZ_FILE_NAME=${LINUX_DISTRO}-${NODE_MODULES_GZ_FILE_PREFIX}-${NODE_MODULES_CACHE_HASH}.tgz

if [[ ${LINUX_DISTRO} == "centos" ]]; then
  echo "Installing AWS CLI..."
  yum update -q
  yum install awscli coreutils jq -q
fi
if [[ ${LINUX_DISTRO} == "debian" ]]; then
  echo "Installing AWS CLI..."
  sudo apt-get update
  sudo apt-get install -y awscli coreutils jq
fi
if [[ ${LINUX_DISTRO} == "alpine" ]]; then
  echo "Installing AWS CLI..."
  apk update
  apk add aws-cli coreutils jq
fi

function find_cache {
  aws s3api list-objects \
    --bucket ${AWS_S3_BUCKET_NAME} \
    --query 'reverse(sort_by(Contents, &LastModified))[].{Key: Key}' \
    --output text | grep "$1" | head -1
}

NODE_MODULES_CACHE=$(find_cache ${NODE_MODULES_GZ_FILE_NAME})
LASTEST_NODE_MODULES_CACHE=$(find_cache ${LINUX_DISTRO}-${NODE_MODULES_GZ_FILE_PREFIX})
CURRENT_CACHE_FILE=${NODE_MODULES_GZ_FILE_NAME}

if [[ ! -z "${NODE_MODULES_CACHE}" ]]; then
  echo "Downloading node_modules cache..."
  aws s3 cp s3://${AWS_S3_BUCKET_NAME}/${NODE_MODULES_GZ_FILE_NAME} ${NODE_MODULES_GZ_FILE_NAME} --quiet || echo ${NODE_MODULES_GZ_FILE_NAME} "does not exist on S3"
  CURRENT_CACHE_FILE=${NODE_MODULES_GZ_FILE_NAME}
elif [[ ! -z "${LASTEST_NODE_MODULES_CACHE}" ]]; then
  echo "Downloading latest node_modules cache..."
  aws s3 cp s3://${AWS_S3_BUCKET_NAME}/${LASTEST_NODE_MODULES_CACHE} ${LASTEST_NODE_MODULES_CACHE} --quiet || echo ${LASTEST_NODE_MODULES_CACHE} "does not exist on S3"
  CURRENT_CACHE_FILE=${LASTEST_NODE_MODULES_CACHE}
else
  echo "Can't find a node_modules cache"
  CURRENT_CACHE_FILE=no-cache
fi

if [[ -f "${CURRENT_CACHE_FILE}" ]]; then
  echo "Using ${CURRENT_CACHE_FILE} as node_modules cache..."
  echo "Extracting node_modules..."
  tar -zxf ${CURRENT_CACHE_FILE}
  rm ${CURRENT_CACHE_FILE}
fi

echo "Running yarn install..."
yarn install --no-progress --prefer-offline

if [[ ${NODE_MODULES_GZ_FILE_NAME} != ${CURRENT_CACHE_FILE} ]]; then
  echo "Compressing node_modules..."
  NODE_MODULES_DIRS=$(yarn --silent workspaces info | jq --raw-output '.[].location' | xargs printf -- './%s/node_modules ')

  # Ensuring every dir exists
  for dir in ${NODE_MODULES_DIRS}; do
    mkdir -p ${dir}
  done

  rm -rf ./node_modules/.cache/nx/d
  tar -zcf ${NODE_MODULES_GZ_FILE_NAME} ./node_modules ${NODE_MODULES_DIRS}

  echo "Uploading node_modules cache..."
  aws s3 cp ${NODE_MODULES_GZ_FILE_NAME} s3://${AWS_S3_BUCKET_NAME}/${NODE_MODULES_GZ_FILE_NAME} --quiet
  rm ${NODE_MODULES_GZ_FILE_NAME}
else
  echo "Skipping node_modules cache upload as it's unchanged..."
fi
