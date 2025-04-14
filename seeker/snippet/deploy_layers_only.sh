#date: 2025-04-14T17:12:23Z
#url: https://api.github.com/gists/e22fbf8f3c99d279f40f4dbb09f860de
#owner: https://api.github.com/users/agomez-arine

#!/bin/bash
set -e

# ==============================================================================
#
# Requirements:
#   - AWS CLI:     Install via `brew install awscli`
#   - jq:          Install via `brew install jq`
#   - zip:         Preinstalled on macOS, or `brew install zip`
#
# Optional:
#   - parallel:    Install via `brew install parallel` to speed up layer updates
#
# Setup:
#   - Ensure you have AWS credentials configured:
#       `aws configure` (uses ~/.aws/credentials)
#
# What this script does:
#   - Builds the arine_api lambda layer
#   - Packages the layer into a .zip file and stores in S3 due to it's large file size
#   - Publishes the layer to each AWS Lambda using Lambda Layers
#
# Run from project root:
#   ./deploy_layers_only.sh -s agomez-dev -p
# ==============================================================================

# START TIME
SECONDS=0

# --- Config ---
DEPLOY_DIR=$PWD
LAYER_NAME=""
S3_KEY=""
S3_BUCKET="arine-api-layers"
RUNTIMES="python3.10"
ARCHITECTURES="x86_64"

PARALLEL=false
# PARALLEL_PROCESSES=$(($(sysctl -n hw.logicalcpu) / 2)) # uses half of system cores
PARALLEL_PROCESSES=$(sysctl -n hw.logicalcpu) # uses all of system cores

usage() {
  echo "Usage:"
  echo "  $0 -s <STACK_NAME> [-p]"
  echo ""
  echo "Options:"
  echo "  -h        show help menu"
  echo "  -s        stack name"
  echo "  -p        parallelize"
  echo ""
  exit 0
}

while getopts "hs:p" flag; do
  case "${flag}" in
  h)
    usage
    exit 0
    ;;
  s) STACK_NAME="${OPTARG}" ;;
  p) PARALLEL=true ;;
  *)
    echo ""
    usage
    exit 1
    ;;
  esac
done
# No options given
if [ $OPTIND -eq 1 ]; then
  usage
  exit 1
fi
# Missing required options
if [ -z "$STACK_NAME" ]; then
  echo "$0: option required -- s"
  echo ""
  usage
  exit 1
fi

# set the variable name based on flag
LAYER_NAME="${STACK_NAME}_arine_api"
S3_KEY="${STACK_NAME}/layer.zip"

build_layer_package() {
  cd "$DEPLOY_DIR/layers/arine_api"
  echo "--> Building arine_api layer"
  ./build_layer.sh

  cd $DEPLOY_DIR
  cp -r arine_api_utils/ layers/arine_api/python/arine_api_utils/

  # # --- zip the layer ---
  rm -f layers/arine_api/layer.zip # cleanup old/existing zip
  echo "--> Zipping arine_api layer and uploading to S3"
  cd "$DEPLOY_DIR/layers/arine_api"
  zip -rq -9 layer.zip python
  # need to upload to s3 due to large size
  aws s3 cp layer.zip "s3://${S3_BUCKET}/${S3_KEY}"

  cd $DEPLOY_DIR
}

# --- Build the layer ---
echo "Building layers"

build_layer_package

# echo "current working directory: $DEPLOY_DIR"

# --- Publish the new layer ---
# docs:
# https://docs.aws.amazon.com/cli/latest/reference/lambda/publish-layer-version.html
echo "--> Publishing new layer..."
# publishing the layer and then grabbing the new layer ARN with the updated version number
# and assigning it to a variable so we can use in the next section

# note: We pass in runtimes and architectures for easy, future extensibility when migrating to ARM64
# and/or Python 3.12+. We also pass the region in case anyone decides to test outside their default region.
# The default region in our AWS profile should be us-east-1
LAYER_ARN=$(aws lambda publish-layer-version \
  --layer-name "$LAYER_NAME" \
  --description "arine_utils" \
  --content S3Bucket="${S3_BUCKET}",S3Key="${S3_KEY}" \
  --compatible-runtimes $RUNTIMES \
  --compatible-architectures $ARCHITECTURES \
  --region "us-east-1" \
  --query 'LayerVersionArn' \
  --output text)
echo "Layer published: $LAYER_ARN"
echo ""

echo "--> Pointing functions to new layer..."

# docs:
# https://docs.aws.amazon.com/cli/latest/reference/lambda/list-functions.html
# 1. list all lambda functions in json to pipe to jq
# 2. set the output to be a raw and pass in the stack name to jq
# 3. check the array elements in the Functions[] json
# 4. filter for functions that have a "Layers" key
#    and has an existing layer specific to your stack
# 5. extracts the name of the function
if [ "$PARALLEL" == true ]; then
  # PARALLEL
  # Must have GNU's `parallel` installed - otherwise use the SEQUENTIAL implementation
  export LAYER_ARN # !!IMPORTANT!!: we need to "export" this var to give process subshells access to this var
  aws lambda list-functions --output json | jq -r --arg STACK_NAME "$STACK_NAME" '
    .Functions[]
    | select(.Layers != null and any(.Layers[]; .Arn | test($STACK_NAME)))
    | .FunctionName
  ' | parallel -j $PARALLEL_PROCESSES '
    echo "Updating: {}" &&
    aws lambda update-function-configuration \
      --function-name "{}" \
      --layers "${LAYER_ARN}" \
      --region us-east-1 \
      --no-cli-pager
  '
else
  # SEQUENTIAL
  aws lambda list-functions --output json | jq -r --arg STACK_NAME "$STACK_NAME" '
    .Functions[]
    | select(.Layers != null and any(.Layers[]; .Arn | test($STACK_NAME)))
    | .FunctionName
  ' | while read FUNC_NAME; do
    echo "Updating: $FUNC_NAME"
    aws lambda update-function-configuration \
      --function-name "$FUNC_NAME" \
      --layers "$LAYER_ARN" \
      --region us-east-1 \
      --no-cli-pager
  done
fi

echo ""
echo "✅ Finished successfully in $SECONDS seconds"
