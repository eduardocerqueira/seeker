#date: 2025-08-11T17:10:53Z
#url: https://api.github.com/gists/72fdf58a0ac67c57082bcd4ac9c4862e
#owner: https://api.github.com/users/pradip-spring

#!/usr/bin/env bash
# export_lambdas.sh — Export AWS Lambda configs to JSON bundles (one file per function).
# Requirements: awscli v2, jq (both available in AWS CloudShell).
# Config via env:
#   OUTDIR=./lambda_export_<timestamp>
#   REGIONS="us-east-1 us-west-2"  (space-separated)
#   ALL_REGIONS=1                  (override REGIONS; scans all regions)
#   INCLUDE_S3=1                   (scan S3 bucket notifications referencing each function)
#   INCLUDE_LOGS=1                 (include CloudWatch Logs info)

set -euo pipefail

command -v aws >/dev/null || { echo "aws CLI not found" >&2; exit 1; }
command -v jq >/dev/null  || { echo "jq not found" >&2; exit 1; }

TS="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
OUTDIR="${OUTDIR:-"./lambda_export_${TS//:/}" }"
INCLUDE_S3="${INCLUDE_S3:-0}"
INCLUDE_LOGS="${INCLUDE_LOGS:-0}"

mkdir -p "$OUTDIR"

current_region() {
  local r
  r="$(aws configure get region || true)"
  if [[ -z "${r:-}" ]]; then
    r="${AWS_REGION:-${AWS_DEFAULT_REGION:-us-east-1}}"
  fi
  echo "$r"
}

regions() {
  if [[ "${ALL_REGIONS:-0}" = "1" ]]; then
    aws ec2 describe-regions --query 'Regions[].RegionName' --output text | tr '\t' '\n' | sort
  elif [[ -n "${REGIONS:-}" ]]; then
    # space or comma separated are both fine
    echo "$REGIONS" | tr ',' ' ' | xargs -n1
  else
    current_region
  fi
}

aws_or_null() {
  # usage: aws_or_null <service> <operation> [args...]
  set +e
  local out
  out=$(aws "$@" 2>/dev/null)
  local rc=$?
  set -e
  if [[ $rc -ne 0 || -z "${out:-}" ]]; then
    echo null
  else
    echo "$out"
  fi
}

eventbridge_rules_obj() {
  # returns {"EventBridgeRuleNames":[...]} or null on error
  local region="$1" arn="$2"
  local resp
  resp="$(aws_or_null events list-rule-names-by-target --region "$region" --target-arn "$arn")"
  if [[ "$resp" == "null" ]]; then
    echo null
    return
  fi
  echo "$resp" | jq -c '{EventBridgeRuleNames: (.RuleNames // [])}'
}

function_urls_obj() {
  local region="$1" fn="$2"
  local resp
  resp="$(aws_or_null lambda list-function-url-configs --region "$region" --function-name "$fn")"
  if [[ "$resp" == "null" ]]; then
    echo null
    return
  fi
  echo "$resp" | jq -c '{FunctionUrlConfigs: (.FunctionUrlConfigs // [])}'
}

tags_obj() {
  local region="$1" arn="$2"
  local resp
  resp="$(aws_or_null lambda list-tags --region "$region" --resource "$arn")"
  if [[ "$resp" == "null" ]]; then
    echo null
    return
  fi
  echo "$resp" | jq -c '{Tags: (.Tags // {})}'
}

logs_obj() {
  local region="$1" fn="$2"
  local group="/aws/lambda/$fn"
  local meta subs info exists
  meta="$(aws_or_null logs describe-log-groups --region "$region" --log-group-name-prefix "$group" --limit 1)"
  if [[ "$meta" == "null" ]]; then
    echo "{\"LogGroup\":{\"Name\":\"$group\",\"Exists\":false}}"
    return
  fi
  exists="$(echo "$meta" | jq -r '(.logGroups // []) | length > 0')"
  if [[ "$exists" == "true" ]]; then
    info="$(echo "$meta" | jq -c '.logGroups[0]')"
    subs="$(aws_or_null logs describe-subscription-filters --region "$region" --log-group-name "$group")"
    if [[ "$subs" != "null" ]]; then
      subs="$(echo "$subs" | jq -c '.subscriptionFilters // []')"
    else
      subs='[]'
    fi
    jq -c -n \
      --arg group "$group" \
      --argjson info "$info" \
      --argjson subs "$subs" \
      '{
        LogGroup:{
          Name:$group,
          Exists:true,
          RetentionInDays:($info.retentionInDays//null),
          KmsKeyId:($info.kmsKeyId//null),
          MetricFilterCount:($info.metricFilterCount//null),
          StoredBytes:($info.storedBytes//null),
          SubscriptionFilters:$subs
        }
      }'
  else
    echo "{\"LogGroup\":{\"Name\":\"$group\",\"Exists\":false}}"
  fi
}

s3_notifications_obj() {
  # Scan all buckets; keep only Lambda notifications that reference $2 (base ARN).
  local arn_base="$1"
  local buckets resp hits out
  buckets="$(aws_or_null s3api list-buckets)"
  if [[ "$buckets" == "null" ]]; then
    echo '{"S3Notifications":[]}'
    return
  fi
  out='[]'
  # iterate bucket names
  echo "$buckets" | jq -r '.Buckets[].Name' | while read -r b; do
    [[ -z "$b" ]] && continue
    resp="$(aws_or_null s3api get-bucket-notification-configuration --bucket "$b")"
    [[ "$resp" == "null" ]] && continue
    hits="$(echo "$resp" | jq -c --arg base "$arn_base" '
      (.LambdaFunctionConfigurations // [])
      | map(select(.LambdaFunctionArn == $base or (.LambdaFunctionArn | startswith($base + ":"))))
      | map({Id, LambdaFunctionArn, Events, Filter})
    ')"
    # append if any hits
    if [[ "$(echo "$hits" | jq 'length')" -gt 0 ]]; then
      out="$(jq -c -n --arg bucket "$b" --argjson acc "$out" --argjson h "$hits" '$acc + [{Bucket:$bucket, Matches:$h}]')"
    fi
  done
  jq -c -n --argjson arr "$out" '{S3Notifications: $arr}'
}

export_fn() {
  local region="$1" fn_name="$2" fn_arn="$3"
  local fn_dir="$OUTDIR/$region"
  mkdir -p "$fn_dir"
  local ts="$TS"

  # Core + fragments
  local get_function eic rc pcc pol esm aliases vers furls tags ebr s3n logs

  get_function="$(aws_or_null lambda get-function --region "$region" --function-name "$fn_name")"
  eic="$(aws_or_null lambda get-function-event-invoke-config --region "$region" --function-name "$fn_name")"
  rc="$(aws_or_null lambda get-function-concurrency --region "$region" --function-name "$fn_name")"
  pcc="$(aws_or_null lambda list-provisioned-concurrency-configs --region "$region" --function-name "$fn_name")"
  pol="$(aws_or_null lambda get-policy --region "$region" --function-name "$fn_name")"
  esm="$(aws_or_null lambda list-event-source-mappings --region "$region" --function-name "$fn_name")"
  aliases="$(aws_or_null lambda list-aliases --region "$region" --function-name "$fn_name")"
  vers="$(aws_or_null lambda list-versions-by-function --region "$region" --function-name "$fn_name")"

  furls="$(function_urls_obj "$region" "$fn_name")"
  tags="$(tags_obj "$region" "$fn_arn")"
  ebr="$(eventbridge_rules_obj "$region" "$fn_arn")"

  if [[ "$INCLUDE_S3" == "1" ]]; then
    s3n="$(s3_notifications_obj "$fn_arn")"
  else
    s3n="null"
  fi
  if [[ "$INCLUDE_LOGS" == "1" ]]; then
    logs="$(logs_obj "$region" "$fn_name")"
  else
    logs="null"
  fi

  # Normalize array-wrapping where useful
  if [[ "$esm" != "null" ]]; then esm="$(echo "$esm" | jq -c '{EventSourceMappings: (.EventSourceMappings // [])}')" ; fi
  if [[ "$aliases" != "null" ]]; then aliases="$(echo "$aliases" | jq -c '{Aliases: (.Aliases // [])}')" ; fi
  if [[ "$vers" != "null" ]]; then vers="$(echo "$vers" | jq -c '{Versions: (.Versions // [])}')" ; fi
  if [[ "$pcc" != "null" ]]; then pcc="$(echo "$pcc" | jq -c '{ProvisionedConcurrencyConfigs: (.ProvisionedConcurrencyConfigs // [])}')" ; fi
  if [[ "$pol" != "null" ]]; then
    # policy string -> object
    pol="$(echo "$pol" | jq -c 'try {Policy: ( .Policy | fromjson )} catch {Policy: .Policy}')"
  fi

  local outfile="$fn_dir/$(echo "$fn_name" | tr -c '[:alnum:]._-' '_' ).json"
  jq -n -c \
    --arg snapshot_ts "$ts" \
    --arg region "$region" \
    --arg function_name "$fn_name" \
    --arg function_arn "$fn_arn" \
    --argjson get_function "${get_function:-null}" \
    --argjson event_invoke_config "${eic:-null}" \
    --argjson reserved_concurrency "${rc:-null}" \
    --argjson provisioned_concurrency_configs "${pcc:-null}" \
    --argjson resource_policy "${pol:-null}" \
    --argjson event_source_mappings "${esm:-null}" \
    --argjson aliases "${aliases:-null}" \
    --argjson versions "${vers:-null}" \
    --argjson function_url_configs "${furls:-null}" \
    --argjson tags "${tags:-null}" \
    --argjson eventbridge_rules "${ebr:-null}" \
    --argjson s3_notifications "${s3n:-null}" \
    --argjson logs "${logs:-null}" \
    '{
      snapshot_ts: $snapshot_ts,
      region: $region,
      function_name: $function_name,
      function_arn: $function_arn,
      get_function: $get_function,
      event_invoke_config: $event_invoke_config,
      reserved_concurrency: $reserved_concurrency,
      provisioned_concurrency_configs: $provisioned_concurrency_configs,
      resource_policy: $resource_policy,
      event_source_mappings: $event_source_mappings,
      aliases: $aliases,
      versions: $versions,
      function_url_configs: $function_url_configs,
      tags: $tags,
      eventbridge_rules: $eventbridge_rules,
      s3_notifications: $s3_notifications,
      logs: $logs
    }' > "$outfile"

  echo "wrote: $outfile" >&2
}

main() {
  local r fnlist
  for r in $(regions); do
    echo "Scanning region: $r" >&2
    fnlist="$(aws_or_null lambda list-functions --region "$r")"
    if [[ "$fnlist" == "null" ]]; then
      echo "  (no access or no functions)" >&2
      continue
    fi
    echo "$fnlist" | jq -c '.Functions[] | {FunctionName, FunctionArn}' | while read -r row; do
      fn="$(echo "$row" | jq -r '.FunctionName')"
      arn="$(echo "$row" | jq -r '.FunctionArn')"
      export_fn "$r" "$fn" "$arn"
    done
  done
  echo "Done. Output under: $OUTDIR" >&2
}

main