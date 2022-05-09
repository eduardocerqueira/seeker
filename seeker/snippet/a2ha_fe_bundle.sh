#date: 2022-05-09T17:14:34Z
#url: https://api.github.com/gists/e985d691628a5c388a455bb6d7545ecf
#owner: https://api.github.com/users/teknofire

#!/usr/bin/env bash

################################################################################
#  This script will generate the necessary airgap bundles and configs for an A2HA 
#  Workstation to deploy a specific version of Automate on the FE nodes.
#
#  By default this will upgrade an A2HA installation with Automate 20220310123121 which is
#  latest version compatabile with A2HA.
#
#  USAGE: 
#          cd /hab/a2_deploy_workspace
#          ./a2ha_fe_bundle.sh
#          automate-cluster-ctl deploy
#
#  To override the specific version run the command like this:
#  AUTOMATE_VERSION=20220329091442 ./automate_fe_bundle.sh
#
#  NOTE: Do not install any version newer than 20220310123121 in A2HA as there is a
#        change in Automate that will prevent the upgrade from completing.
#
#  Author: Will Fisher <will.fisher@progress.com>
################################################################################

set -euo pipefail

VERSION=${AUTOMATE_VERSION:-20220310123121}
A2HA_AIB=frontend-${VERSION}.aib
WS_PATH=${WORKSPACE_PATH:-/hab/a2_deploy_workspace}

echo "Building airgap bundle for ${VERSION}"

/tmp/chef-automate airgap bundle create --version ${VERSION} ${WS_PATH}/terraform/transfer_files/$A2HA_AIB
md5sum ${WS_PATH}/terraform/transfer_files/$A2HA_AIB > ${WS_PATH}/terraform/transfer_files/${A2HA_AIB}.md5
echo "frontend_aib_dest_file = \"/var/tmp/${A2HA_AIB}\"
frontend_aib_local_file = \"${A2HA_AIB}\"
" > ${WS_PATH}/terraform/a2ha_aib_fe.auto.tfvars