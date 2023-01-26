#date: 2023-01-26T16:47:42Z
#url: https://api.github.com/gists/b43ba89194761fe92a749316779cc8ea
#owner: https://api.github.com/users/ormergi

# This script enable faster development iterations in Kubevirt
# when working on a change that might affect backward compatibility
# upgrades and legacy VMs (a VM that existed before an upgrade) disruption.
#
# Environment variables:
# PREV_VER_BRANCH:      Previous version branch where previous version Kubevirt is built from.
# PREV_VER_TAG:         Previous version Kubevirt images tag.
# PREV_VER_REGISTRY:    Previous version Kubevirt images registry.
# TARGET_VER_BRANCH:    Target version branch name where target version Kubevirt is built from.
# TARGET_VER_TAG:       Target version Kubevirt images tag.
# TARGET_VER_REGISTRY:  Target version Kubevirt images registry.
# 
# Flags: 
#   --build-previous-version:   Checkout to the branch of the version Kubevirt upgrading from  
#                               and build & push the images to the local registry.
#    
#   --build-target-version:     Checkout to the branch of the version Kubevirt upgrading to  
#                               and build & push the images to the local registry.
#                               Note: changes from current branch are stashed 
#
#   --functest:                 Run kubevirt e2e tests.
#                               All necessary flags are passed to ensure the correct tags are used in tests.
#
#   --cluster-up:               Spin up ephemeral cluster using github.com/kubevirt/kubevirtci.
#   
#   --deploy-target-version:    Deploy Kubevirt that was build from the target version branch.
#
#   --deploy-previous-version:  Deploy Kubevirt that was build from the previous version branch.
#                               Note: The Kubevirt operator is deployed from the target version branch
#                                     in order to have the latest Kubevirt CRDs.
#
# Example:
# Test Kubevirt upgrade between two versions that were built from local branches:
# export TARGET_VER_BRANCH=my-cool-feature
# export TARGET_VER_TAG=latest
# export TARGET_VER_REGISTRY=registry:5000/kubevirt
# export PREV_VER_BRANCH=main
# export PREV_VER_TAG=devel
# export PREV_VER_REGISTRY=registry:5000/kubevirt
#
# export KUBEVIRT_E2E_FOCUS="<upgrade test key>"
# ./dev-iterate.sh --build-previous-version --build-target-version --deploy-target-version --functest
# 
# In case the test expectes the previous version to be deployed:
# ./dev-iterate.sh --build-previous-version --build-target-version --deploy-previous-version --functest
#
#!/bin/bash

set -e

CLUSTER_UP=false
BUILD_PREVIOUS_VERSION_KUBEVIRT=false
BUILD_TARGET_VERSION_KUBEVIRT=false
DEPLOY_PREVIOUS_VERSION_KUBEVIRT=false
DEPLOY_TARGET_VERSION_KUBEVIRT=false
FUNCTEST=true

options=$(getopt --options "" \
    --long cluster-up,build-previous-version,build-target-version,deploy-previous-version,deploy-target-version,functest,help\
    -- "${@}")
eval set -- "$options"
while true; do
    case "$1" in
     --cluster-up)
        CLUSTER_UP=true
        ;;
    --build-previous-version)
        BUILD_PREVIOUS_VERSION_KUBEVIRT=true
        ;;
    --build-target-version)
        BUILD_TARGET_VERSION_KUBEVIRT=true
        ;;
    --deploy-previous-version)
        DEPLOY_PREVIOUS_VERSION_KUBEVIRT=true
        ;;
    --deploy-target-version)
        DEPLOY_TARGET_VERSION_KUBEVIRT=true
        ;;
    --functest)
        FUNCTEST=true
        ;;
    --help)
        set +x
        echo "$0 [--cluster-up] [--build-previous-version] [--build-target-version] [--deploy-previous-version] [--deploy-target-version] [--functest]"
        exit
        ;;
    --)
        shift
        break
        ;;
    esac
    shift
done

if ! kubectl cluster-info; then
  echo "failed to interact with the cluster API"
  exit 1
fi

if ${CLUSTER_UP}; then
  export KUBEVIRT_PROVIDER="k8s-1.25"
  export KUBEVIRT_NUM_NODES="2"
  export KUBEVIRT_NUM_SECONDARY_NICS="1"
  export KUBEVIRT_WITH_CNAO="true"
  export KUBEVIRT_DEPLOY_ISTIO="true"
  export KUBEVIRT_DEPLOY_CDI="false"
  export KUBECONFIG=$(./cluster-up/kubeconfig.sh)

  make cluster-up
fi

PREV_VER_BRANCH="${PREV_VER_BRANCH:-pr6852}"
PREV_VER_TAG="${PREV_VER_TAG:-hotplug}"
PREV_VER_REGISTRY="${PREV_VER_REGISTRY:-registry:5000/kubevirt}"

TARGET_VER_BRANCH="${TARGET_VER_BRANCH:-pod-iface-names}"
TARGET_VER_TAG="${TARGET_VER_TAG:-latest}"
TARGET_VER_REGISTRY="${TARGET_VER_REGISTRY:-egistry:5000/kubevirt}"

readonly PREV_KV_MANIFESTS="./prev-kv-manifests"
readonly GIT_STASH_ENTRY="iterate"

patch_kubevirt_cr() {
    set -r path=$1
    set -r tag=$2
    set -r registry=$3

    # set Kubevirt workload update strategy to LiveMigrate, VMs will migrate as soon as an upgrade completes
    sed -i "s?workloadUpdateStrategy: {}?workloadUpdateStrategy:\n    workloadUpdateMethods:\n    - LiveMigrate?g" ${path}
    # specify previous version explicitly to deploy Kubevirt from previous version
    sed -i "s?spec:?spec:\n  imageTag: ${tag}\n  imageRegistry: ${registry}?g" ${path}
    # ensure the latest images digest is used by setting kubevirt deployments imagePullPolicy to always
    sed -i "s?imagePullPolicy: IfNotPresent?imagePullPolicy: Always?g" ${path}
    # raise set kubevirt components log verbosity
    sed -i "s?developerConfiguration:?developerConfiguration:\n      logVerbosity:\n        virtHandler: 4\n        virtLauncher: 4?g" ${path}
}

if ${BUILD_PREVIOUS_VERSION_KUBEVIRT}; then
  current_branch=$(git status | grep -i -Po "On branch \K.*")
  if [ "${current_branch}" != "${PREV_VER_BRANCH}" ]; then
    # stash current changes
    if git status -s | grep M; then
      git stash save ${GIT_STASH_ENTRY}
    fi
    # switch to Kubevirt previous version branch
    git checkout ${PREV_VER_BRANCH}
  fi
  
  # build Kubevirt from previous version branch and push to local registry
  DOCKER_TAG=${PREV_VER_TAG} DOCKER_TAG_ALT=${PREV_VER_TAG} make cluster-build manifests
  
  # keep previous version manifests from deployment
  mkdir -p ${PREV_KV_MANIFESTS}
  cp -r "_out"/* "${PREV_KV_MANIFESTS}"
fi

if ${BUILD_TARGET_VERSION_KUBEVIRT}; then
  current_branch=$(git status | grep -i -Po "On branch \K.*")
  if [ "${current_branch}" != "${TARGET_VER_BRANCH}" ]; then
    # switch to development branch
    git checkout ${TARGET_VER_BRANCH}
    # pop development branch changes
    if git stash list | grep "stash@{0}" | grep "${GIT_STASH_ENTRY}"; then
      git stash pop stash@{0}
    fi
  fi
  
  # build Kubevirt from development branch and push to local registry
  DOCKER_TAG=${TARGET_VER_TAG} DOCKER_TAG_ALT=${TARGET_VER_TAG} make cluster-build manifests
fi

if ${DEPLOY_PREVIOUS_VERSION_KUBEVIRT}; then
  # patch Kubevirt CR
  PATCHED_KUBEVIRT_CR="${PREV_KV_MANIFESTS}/manifests/release/kubevirt-cr-patched.yaml"
  cp ${PREV_KV_MANIFESTS}/manifests/release/kubevirt-cr.yaml ${PATCHED_KUBEVIRT_CR}
  patch_kubevirt_cr ${PATCHED_KUBEVIRT_CR} ${PREV_VER_TAG} ${PREV_VER_REGISTRY}

  # deploy Kubevirt operator from development version (if previous version operator is deployed we wont get the latest CRDs)
  kubectl apply -f _out/manifests/release/kubevirt-operator.yaml

  # deploy Kubevirt CR previous version
  kubectl apply -f ${PATCHED_KUBEVIRT_CR}

  # wait for Kubevirt to be ready
  kubectl wait kubevirt -n kubevirt kubevirt --for condition=Available=True --timeout 5m
fi

if ${DEPLOY_TARGET_VERSION_KUBEVIRT}; then
  # patch Kubevirt CR
  PATCHED_KUBEVIRT_CR="./kubevirt-cr-patched.yaml"
  cp _out/manifests/release/kubevirt-cr.yaml ${PATCHED_KUBEVIRT_CR}
  patch_kubevirt_cr ${PATCHED_KUBEVIRT_CR} ${TARGET_VER_TAG} ${TARGET_VER_REGISTRY}

  # deploy Kubevirt operator from development version (if previous version operator is deployed we wont get the latest CRDs)
  kubectl apply -f _out/manifests/release/kubevirt-operator.yaml

  # deploy Kubevirt CR previous version
  kubectl apply -f ${PATCHED_KUBEVIRT_CR}

  # wait for Kubevirt to be ready
  kubectl wait kubevirt -n kubevirt kubevirt --for condition=Available=True --timeout 5m
fi

if ${FUNCTEST}; then
  # run operator test:
  # 1. Create a VM with secondary NIC
  # 2. Upgrade kubevirt to development branch version.
  # 3. Assert that VM been migrated successfully following the upgrade
  # 4. Migrate the VM again
  # 5. Assert that VM been migrated successfully
  args=()
  args+=("-container-prefix=${TARGET_VER_REGISTRY}")
  args+=("-container-tag=${TARGET_VER_TAG}")
  args+=("-utility-container-prefix=${TARGET_VER_REGISTRY}")
  args+=("-utility-container-tag=${TARGET_VER_TAG}")

  args+=("-previous-release-registry=${PREV_VER_REGISTRY}")
  args+=("-previous-release-tag=${PREV_VER_TAG}")
  args+=("-previous-utility-container-registry=${PREV_VER_REGISTRY}")
  args+=("-previous-utility-container-tag=${PREV_VER_TAG}")

  args+=("--ginkgo.v --ginkgo.progress")
  args="${args[*]}"

  KUBEVIRT_FUNC_TEST_SUITE_ARGS="${args}" make functest
  #DOCKER_TAG=${TARGET_VER_TAG} DOCKER_TAG_ALT=${TARGET_VER_TAG} PREVIOUS_RELEASE_REGISTRY=${PREV_VER_REGISTRY} PREVIOUS_RELEASE_TAG=${PREV_VER_TAG} KUBEVIRT_FUNC_TEST_SUITE_ARGS="${args}" make functest
fi