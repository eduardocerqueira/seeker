#date: 2024-09-17T17:00:19Z
#url: https://api.github.com/gists/3d35243c3b968b9d1d883680ec0ea2fe
#owner: https://api.github.com/users/darksinge

#!/usr/bin/env bash

set -e

NO_CACHE=0
CLEAN=0

AWS_PROFILE=""
AWS_REGION="us-east-1"

help() {
    cat << EOF
Usage: $(basename "$0") [OPTIONS]

This script helps you find and open an AWS Lambda function in the AWS Console.

Options:
  -p, --profile PROFILE   Specify the AWS profile to use
  -r, --region  REGION    Specify the AWS region to use
  --no-cache              Disable caching of AWS resources
  --clean                 Clear out cached results
  -h, --help              Display this help message and exit

The script will prompt you to:
1. Choose or enter a stage (dev/stage/prod/other)
2. Select a CloudFormation stack
3. Choose a Lambda function from the selected stack

It will then open the chosen Lambda function in your default web browser.

Note: This script requires the following tools to be installed:
- AWS CLI
- jq
- gum (for interactive prompts)
EOF
}

if ! gum -v >/dev/null 2>&1; then
  echo "The 'gum' command was not found."
  echo "Visit https://github.com/charmbracelet/gum for installation instructions."
  exit 1
fi

while [[ $# -gt 0 ]]; do
  case $1 in
    -p|--profile)
      export AWS_PROFILE="$2"
      shift 2
      ;;
    --no-cache)
      export NO_CACHE=1
      shift
      ;;
    --region)
      AWS_REGION="$2"
      shift 2
      ;;
    --clean)
      CLEAN=1
      ;;
    -h|--help)
      help
      exit 0
      ;;
    *)
      help
      exit 1
      ;;
  esac
done

CACHE_DIR="$HOME/.cache/find-fn"
if [ ! -d "$CACHE_DIR" ]; then
  mkdir -p "$CACHE_DIR"
fi

if [ $CLEAN -eq 1 ]; then
  rm -rf "$CACHE_DIR" >/dev/null 2>&1
  mkdir -p "$CACHE_DIR"
  exit 0
fi

STAGE=$(gum choose "dev" "stage" "prod" "other")
if [ "$STAGE" == "other" ]; then
  STAGE=$(gum input --placeholder "stage name?")
fi

STACKS_LIST_CACHE="$CACHE_DIR/$STAGE-stacks"

function _make_temp() {
  type="$1"
  fcache="$CACHE_DIR/$STAGE-$type"

  if [ $NO_CACHE -eq 1 ]; then
    echo "$(mktemp)"
    return 0
  fi

  local tmp=""
  if [ -f "$fcache" ]; then
    tmp=$(cat "$fcache")
  fi

  if [ ! -f "$tmp" ]; then
    tmp=$(mktemp)
    echo "$tmp" > "$fcache"
  else
    tmp=$(cat "$fcache")
  fi

  echo "$tmp"
}

function make_temp() {
  set +e
  echo $(_make_temp "$1")
  set -e
}

stack_list_cache=$(make_temp "stacks")
if [ -f "$stack_list_cache" ]; then
  STACKS=$(cat "$stack_list_cache")
fi

if [ -z "$STACKS" ]; then
  STACKS=$(gum spin --spinner dot --title 'Fetching stacks' --show-output -- \
            aws cloudformation list-stacks \
            --query "StackSummaries[?starts_with(StackName, '$STAGE-certifications-service-')].StackName" \
            --output json)

  echo "$STACKS" > "$stack_list_cache"
fi

PREFIX="$STAGE-certifications-service-"
STACK_NAME=$(gum choose $(echo "$STACKS" | jq -r '.[]' | sed "s/$PREFIX//"))
STACK_NAME="$PREFIX$STACK_NAME"

resource_cache=$(make_temp "$STACK_NAME-resources")
if [ -f "$resource_cache" ]; then
  RESOURCES=$(cat "$resource_cache")
fi

if [ -z "$RESOURCES" ]; then
  RESOURCES=$(gum spin --spinner dot --title 'Fetching resources' --show-output -- \
              aws cloudformation list-stack-resources --stack-name "$STACK_NAME" \
              --output json)
  echo "$RESOURCES" > "$resource_cache"
fi

RESOURCES=$(cat "$resource_cache" | jq '.StackResourceSummaries')

LOGICAL_ID=$(echo "$RESOURCES" | jq -r '.[] | select(.ResourceType == "AWS::Lambda::Function") | .LogicalResourceId' | gum filter)
PHYSICAL_ID=$(echo "$RESOURCES" | jq -r ".[] | select(.LogicalResourceId == \"$LOGICAL_ID\") | .PhysicalResourceId")

if [ -n "$PHYSICAL_ID" ]; then
  open "https://$AWS_REGION.console.aws.amazon.com/lambda/home?region=$AWS_REGION#/functions/$PHYSICAL_ID?tab=monitor"
fi
