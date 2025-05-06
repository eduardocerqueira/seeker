#date: 2025-05-06T16:58:27Z
#url: https://api.github.com/gists/8dd7148b4aa4d4673396db0c77b4e75b
#owner: https://api.github.com/users/l0g1x

#!/bin/bash

# Script to fetch CloudWatch logs.
# - Supports CSV (default) or JSONL output formats.
# - If a service type (--replication-manager or --view-syncer) is specified,
#   it fetches logs for that service with the lookback period.
# - If no service type is specified, the lookback period is the first (or second after format flag)
#   argument, and logs are fetched for BOTH replication-manager and view-syncer.
#
# Usage (specific service):
#   ./get-cw-logs-unified.sh [--csv|--json] <service_type> <lookback_duration>
#   Example: ./get-cw-logs-unified.sh --csv --replication-manager 30M
#            ./get-cw-logs-unified.sh --json --view-syncer 2H
#            ./get-cw-logs-unified.sh --replication-manager 1D (defaults to CSV)
#
# Usage (both services):
#   ./get-cw-logs-unified.sh [--csv|--json] <lookback_duration>
#   Example: ./get-cw-logs-unified.sh --csv 1D
#            ./get-cw-logs-unified.sh 2m (defaults to CSV, fetches for 2 months)

set -e # Exit immediately if a command exits with a non-zero status.

# --- Global Configuration ---
PROFILE='your-aws-config-profile-staging'

# --- Function to Fetch Logs for a Service ---
fetch_logs_for_service() {
  local SERVICE_TYPE_FLAG=$1
  local LOOKBACK_DURATION_ARG=$2
  local OUTPUT_FORMAT_FUNC=$3 # New parameter for output format
  local SERVICE_NAME=""
  local LOG_GROUP=""
  local FILTER_OUT_PATTERN=""
  local OUTPUT_FILE_PREFIX=""

  if [ "$SERVICE_TYPE_FLAG" == "--replication-manager" ]; then
    SERVICE_NAME="replication-manager"
    # UPDATE THIS TO YOUR LOG GROUP 'sst/cluster/.............../replication-manager/replication-manager'
    LOG_GROUP=
    FILTER_OUT_PATTERN='Executed statement'
    OUTPUT_FILE_PREFIX="replication-manager-filtered"
  elif [ "$SERVICE_TYPE_FLAG" == "--view-syncer" ]; then
    SERVICE_NAME="view-syncer"
    # UPDATE THIS TO YOUR LOG GROUP 'sst/cluster/.............../view-syncer/view-syncer'
    LOG_GROUP=
    FILTER_OUT_PATTERN="" # No filtering for view-syncer
    OUTPUT_FILE_PREFIX="view-syncer-allstreams"
  else
    echo "Error (internal): Invalid service type flag '$SERVICE_TYPE_FLAG' passed to function."
    return 1 # Use return for function error, set -e will handle script exit if needed
  fi

  echo "--- Processing $SERVICE_NAME (Output: $(echo "$OUTPUT_FORMAT_FUNC" | tr '[:lower:]' '[:upper:]')) ---"

  # --- Input Validation for Lookback Duration (within function) ---
  if ! [[ "$LOOKBACK_DURATION_ARG" =~ ^[0-9]+[MmHD]$ ]]; then
      echo "Error for $SERVICE_NAME: Invalid lookback format '$LOOKBACK_DURATION_ARG'. Use numbers followed by M (Minutes), m (months), H (Hours), or D (Days)."
      echo "Example: 30M, 2m, 2H, 1D"
      return 1
  fi
  # --------------------------------------------

  # --- Calculate Start Time ---
  local LOOKBACK_NUM=$(echo "$LOOKBACK_DURATION_ARG" | sed 's/.$//')
  local LOOKBACK_UNIT=$(echo "$LOOKBACK_DURATION_ARG" | sed 's/.*\(.\)$/\1/')
  local START_TIME_MS
  local BSD_DATE_UNIT_SUFFIX=""
  if [ "$LOOKBACK_UNIT" == "M" ]; then BSD_DATE_UNIT_SUFFIX="M";
  elif [ "$LOOKBACK_UNIT" == "m" ]; then BSD_DATE_UNIT_SUFFIX="m";
  elif [ "$LOOKBACK_UNIT" == "H" ]; then BSD_DATE_UNIT_SUFFIX="H";
  elif [ "$LOOKBACK_UNIT" == "D" ]; then BSD_DATE_UNIT_SUFFIX="d";
  fi
  START_TIME_MS=$(date -v -"$LOOKBACK_NUM""$BSD_DATE_UNIT_SUFFIX" +%s000 2>/dev/null)
  if [ $? -ne 0 ]; then
      echo "Warning for $SERVICE_NAME: Failed to calculate start time with BSD 'date -v -${LOOKBACK_NUM}${BSD_DATE_UNIT_SUFFIX}'. Trying GNU date..."
      local LOOKBACK_GNU_STR="$LOOKBACK_NUM"
      if [ "$LOOKBACK_UNIT" == "M" ]; then LOOKBACK_GNU_STR+=" minutes";
      elif [ "$LOOKBACK_UNIT" == "m" ]; then LOOKBACK_GNU_STR+=" months";
      elif [ "$LOOKBACK_UNIT" == "H" ]; then LOOKBACK_GNU_STR+=" hours";
      elif [ "$LOOKBACK_UNIT" == "D" ]; then LOOKBACK_GNU_STR+=" days";
      fi
      START_TIME_MS=$(date -d "$LOOKBACK_GNU_STR ago" +%s000 2>/dev/null)
      if [ $? -ne 0 ]; then
          echo "Error for $SERVICE_NAME: Both BSD and GNU date commands failed. Cannot determine start time for lookback '$LOOKBACK_DURATION_ARG'."
          return 1
      fi
      echo "GNU date fallback successful for $SERVICE_NAME."
  fi
  echo "Fetching logs for $SERVICE_NAME starting from $(date -r $((START_TIME_MS / 1000)))"
  # --------------------------

  # --- Directory and File Setup ---
  # Calculate directory name based on START_TIME_MS in UTC
  local OUTPUT_DIR_NAME
  OUTPUT_DIR_NAME=$(TZ=UTC date -r $((START_TIME_MS / 1000)) +'%Y-%m-%d_%I:%M:%S%p_UTC')
  if [ -z "$OUTPUT_DIR_NAME" ]; then # Basic check in case date command failed silently
      echo "Error for $SERVICE_NAME: Failed to generate output directory name from START_TIME_MS."
      return 1
  fi

  # Create the output directory
  mkdir -p "$OUTPUT_DIR_NAME"
  if [ ! -d "$OUTPUT_DIR_NAME" ]; then
      echo "Error for $SERVICE_NAME: Failed to create output directory '$OUTPUT_DIR_NAME'."
      return 1
  fi

  local OUTPUT_FILE_EXT="jsonl"
  if [ "$OUTPUT_FORMAT_FUNC" == "csv" ]; then
    OUTPUT_FILE_EXT="csv"
  fi
  # Updated OUTPUT_FILE to include the new directory path
  local OUTPUT_FILE="${OUTPUT_DIR_NAME}/${OUTPUT_FILE_PREFIX}-${LOOKBACK_DURATION_ARG}-$(date +%Y%m%d%H%M%S).${OUTPUT_FILE_EXT}"
  local TEMP_OUTPUT="temp_cw_filter_output_$$_${SERVICE_NAME}.json"
  # ------------------

  # Write CSV header if applicable
  if [ "$OUTPUT_FORMAT_FUNC" == "csv" ]; then
    echo "logStreamName,timestamp,ingestionTime,eventId,message.level,message.worker,message.pid,message.component,message.clientGroupID,message.instance,message.message" > "$OUTPUT_FILE"
  fi

  # --- Fetch and Optionally Filter Logs ---
  local NEXT_TOKEN= "**********"
  local PREV_TOKEN= "**********"

  echo "Fetching logs for $SERVICE_NAME from all streams since $LOOKBACK_DURATION_ARG ago..."
  if [ -n "$FILTER_OUT_PATTERN" ]; then
    echo "Filtering out log events for $SERVICE_NAME containing: '$FILTER_OUT_PATTERN'"
  fi

  # Initial fetch
  aws logs filter-log-events --log-group-name "$LOG_GROUP" --start-time "$START_TIME_MS" --profile "$PROFILE" > "$TEMP_OUTPUT"
  if [ $? -ne 0 ]; then
      echo "Error: Initial AWS CLI command failed for $SERVICE_NAME."
      rm -f "$TEMP_OUTPUT" # Clean up partial temp file
      return 1
  fi

  NEXT_TOKEN= "**********"

  # Define the core jq logic for converting a single event to CSV
  JQ_SINGLE_EVENT_TO_CSV_LOGIC='(.message | fromjson?) as $msg_json | [.logStreamName, (.timestamp / 1000 | todateiso8601), (.ingestionTime / 1000 | todateiso8601), .eventId, $msg_json.level // "", $msg_json.worker // "", ($msg_json.pid // ""), $msg_json.component // "", $msg_json.clientGroupID // "", $msg_json.instance // "", $msg_json.message // ""] | @csv'

  # Process and append events
  if [ "$OUTPUT_FORMAT_FUNC" == "csv" ]; then
    if [ -n "$FILTER_OUT_PATTERN" ]; then
      jq -r --arg pattern "$FILTER_OUT_PATTERN" ".events[] | select(.message | contains(\$pattern) | not) | $JQ_SINGLE_EVENT_TO_CSV_LOGIC" "$TEMP_OUTPUT" >> "$OUTPUT_FILE"
    else
      jq -r ".events[] | $JQ_SINGLE_EVENT_TO_CSV_LOGIC" "$TEMP_OUTPUT" >> "$OUTPUT_FILE"
    fi
  else # jsonl
    if [ -n "$FILTER_OUT_PATTERN" ]; then
      jq -c --arg pattern "$FILTER_OUT_PATTERN" '.events[] | select(.message | contains($pattern) | not)' "$TEMP_OUTPUT" >> "$OUTPUT_FILE"
    else
      jq -c '.events[]' "$TEMP_OUTPUT" >> "$OUTPUT_FILE"
    fi
  fi

  # Paginate if needed
  while [ "$NEXT_TOKEN" != "**********"= "$PREV_TOKEN" ]; do
    echo "Fetching next page for $SERVICE_NAME (all streams)..."
    PREV_TOKEN= "**********"
    aws logs filter-log-events --log-group-name "$LOG_GROUP" --profile "$PROFILE" --next-token "$NEXT_TOKEN" > "$TEMP_OUTPUT"
    if [ $? -ne 0 ]; then
          echo "Warning: AWS CLI command failed during pagination for $SERVICE_NAME. Logs might be incomplete."
          break # Exit loop on pagination error
      fi
    NEXT_TOKEN= "**********"
    if [ -n "$FILTER_OUT_PATTERN" ]; then
      jq -c --arg pattern "$FILTER_OUT_PATTERN" '.events[] | select(.message | contains($pattern) | not)' "$TEMP_OUTPUT" >> "$OUTPUT_FILE"
    else
      jq -c '.events[]' "$TEMP_OUTPUT" >> "$OUTPUT_FILE"
    fi
  done
  # ------------------------------------

  # --- Cleanup ---
  rm -f "$TEMP_OUTPUT" # Clean up temporary file, -f to suppress error if already deleted
  # -------------

  echo "--------------------------------------------------"
  if [ -n "$FILTER_OUT_PATTERN" ]; then
    echo "Filtered logs for $SERVICE_NAME (format: $(echo "$OUTPUT_FORMAT_FUNC" | tr '[:lower:]' '[:upper:]')) from last $LOOKBACK_DURATION_ARG saved to: $OUTPUT_FILE"
  else
    echo "All logs for $SERVICE_NAME (format: $(echo "$OUTPUT_FORMAT_FUNC" | tr '[:lower:]' '[:upper:]')) from last $LOOKBACK_DURATION_ARG saved to: $OUTPUT_FILE"
  fi
  echo "--------------------------------------------------"
  return 0 # Success
}
# --- End of Function ---


# --- Main Script Logic ---
OUTPUT_FORMAT_MAIN="csv" # Default to CSV

# Check for output format flag first
if [ "$1" == "--csv" ]; then
  OUTPUT_FORMAT_MAIN="csv"
  shift
elif [ "$1" == "--json" ]; then
  OUTPUT_FORMAT_MAIN="json"
  shift
fi

# Check if jq is installed
if ! command -v jq &> /dev/null; then
    echo "Error: jq is not installed. Please install jq (brew install jq)."
    exit 1
fi

if [ -z "$1" ]; then
  echo "Error: Arguments are required."
  echo "Usage (specific service): $0 [--csv|--json] <service_type> <lookback_duration>"
  echo "  Example: $0 --csv --replication-manager 30M"
  echo "Usage (both services): $0 [--csv|--json] <lookback_duration>"
  echo "  Example: $0 1D"
  exit 1
fi

ARG1_AFTER_FORMAT_SHIFT="$1"
LOOKBACK_DURATION_MAIN=""

if [ "$ARG1_AFTER_FORMAT_SHIFT" == "--replication-manager" ] || [ "$ARG1_AFTER_FORMAT_SHIFT" == "--view-syncer" ]; then
  # Mode: Specific service
  if [ -z "$2" ]; then
    echo "Error: Lookback duration is required when specifying service type '$ARG1_AFTER_FORMAT_SHIFT'."
    echo "Usage: $0 [--csv|--json] $ARG1_AFTER_FORMAT_SHIFT <lookback_duration> (e.g., 30M, 2m, 2H, 1D)"
    exit 1
  fi
  LOOKBACK_DURATION_MAIN="$2"
  fetch_logs_for_service "$ARG1_AFTER_FORMAT_SHIFT" "$LOOKBACK_DURATION_MAIN" "$OUTPUT_FORMAT_MAIN"
else
  # Mode: Potentially both services, or invalid argument
  LOOKBACK_DURATION_MAIN="$ARG1_AFTER_FORMAT_SHIFT" # Assume this is lookback duration

  if ! [[ "$LOOKBACK_DURATION_MAIN" =~ ^[0-9]+[MmHD]$ ]]; then
      echo "Error: Invalid argument '$LOOKBACK_DURATION_MAIN' following format flag (or as first argument)."
      echo "Expected --replication-manager, --view-syncer, or a lookback duration (e.g., 30M, 2m, 2H, 1D)."
      echo "Usage (specific service): $0 [--csv|--json] <service_type> <lookback_duration>"
      echo "Usage (both services): $0 [--csv|--json] <lookback_duration>"
      exit 1
  fi

  echo "Fetching logs for both replication-manager and view-syncer (format: $(echo "$OUTPUT_FORMAT_MAIN" | tr '[:lower:]' '[:upper:]')) with lookback $LOOKBACK_DURATION_MAIN..."
  echo ""

  fetch_logs_for_service "--replication-manager" "$LOOKBACK_DURATION_MAIN" "$OUTPUT_FORMAT_MAIN"
  EXIT_CODE_RM=$?

  echo "" # Add a newline for better separation in console output
  echo "==================================================" # Larger separator
  echo "" # Add a newline for better separation in console output

  fetch_logs_for_service "--view-syncer" "$LOOKBACK_DURATION_MAIN" "$OUTPUT_FORMAT_MAIN"
  EXIT_CODE_VS=$?

  if [ $EXIT_CODE_RM -ne 0 ] || [ $EXIT_CODE_VS -ne 0 ]; then
    echo "One or both log fetching operations failed."
    # set -e would have already exited if a command inside the function failed and returned non-zero
    # This is more for if the function itself returns a failure code but doesn't exit due to `set -e` not being in function scope for `return`
    # However, `set -e` is global, so aws/jq failing *will* exit.
    # This is mostly a safeguard or for future if `set -e` is removed from function.
    # Actually, with `set -e`, if `fetch_logs_for_service` returns 1, the script will exit.
    # So, we might not even reach the second call if the first one fails due to an internal error (like bad date).
    # If `aws` or `jq` fails, `set -e` handles it. If `date` fails and we `return 1`, `set -e` handles it.
    # So, explicit check of $? might be redundant with `set -e`.
  fi
  echo "Completed fetching for both services."

fi

exit 0
fi
  echo "Completed fetching for both services."

fi

exit 0
