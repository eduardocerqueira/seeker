#date: 2023-02-07T17:05:03Z
#url: https://api.github.com/gists/dd4f2d7c4cd85ea4f6f37ad2b1001fbb
#owner: https://api.github.com/users/wendtek

#!/usr/bin/env bash
# Author: Kevin Wendt @wendtek
# This code is licensed under the terms of the MIT license

# Requires jq and curl
# Edit the function_list with whatever functions you want to export
# To get a copyable list of all functions: aws lambda list-functions | jq '.Functions[].FunctionName'

set -euo pipefail

### Function list to edit
function_list=(
  "function_name1"
  "function_name2"
)

### AWS CLI options
aws_opts=""

### Prepare workspace
workdir=$(mktemp -d /tmp/lambda_export.XXXXX)
catch() {
  if [ "$1" != "0" ]; then
    echo "Error $1 occurred on line $2"
  fi

  echo -e "\nRemoving temporary directory"
  rm -rf ${workdir}

  exit $1
}
trap 'catch $? $LINENO' EXIT

### Core logic
pushd ${workdir}
  ### Iterate through lambdas in list
  for function in "${function_list[@]}"; do
    mkdir -p ${function}
    aws ${aws_opts} lambda get-function --function-name ${function} > ${function}/export.json
    code_location=$(jq -r '.Code.Location' ${function}/export.json)
    curl -L -o ${function}/code.zip "${code_location}"
    for arn in $(jq -r '.Configuration.Layers[].Arn' ${function}/export.json) ; do
      layer_arns+=( "${arn}" )
    done
  done

  ### Iterate though dependant layers
  unique_layer_arns=($(echo "${layer_arns[@]}" | tr ' ' '\n' | sort -u | tr '\n' ' '))
  for arn in "${layer_arns[@]}"; do
    mkdir -p ${arn}
    aws ${aws_opts} lambda get-layer-version-by-arn --arn ${arn} > ${arn}/export.json
    code_location=$(jq -r '.Content.Location' ${arn}/export.json)
    curl -L -o ${arn}/code.zip "${code_location}"
  done

  bundle_name="lambda_export_$(date +%Y%m%d_%H%M).tar.gz"
  tar -zcvf ${bundle_name} *
popd

cp ${workdir}/${bundle_name} .
