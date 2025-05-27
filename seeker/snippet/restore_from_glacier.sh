#date: 2025-05-27T17:12:50Z
#url: https://api.github.com/gists/6a2124c4c7e159c47fec8bc6865c0b4b
#owner: https://api.github.com/users/kolaCZek

#!/bin/bash

# Usage: ./restore_from_file_with_args_and_skip.sh <input_file> <bucket> <profile> [<start_line>]
#
# aws s3 ls s3://bucket/prefix/ --recursive --profile profile-name | awk '{print $4}' > /tmp/output.txt


if [[ $# -lt 3 || $# -gt 4 ]]; then
  echo "Usage: $0 <input_file> <bucket> <profile> [<start_line>]"
  exit 1
fi

input_file="$1"
bucket="$2"
profile="$3"
start_line="${4:-1}"
restore_request='{"Days":2,"GlacierJobParameters":{"Tier":"Standard"}}'

if [[ ! -f "$input_file" ]]; then
  echo "Soubor $input_file neexistuje!"
  exit 2
fi

total_lines=$(wc -l < "$input_file")
current_line=0
start_time=$(date +%s)

while IFS= read -r line || [[ -n "$line" ]]; do
  ((current_line++))
  if (( current_line < start_line )); then
    continue
  fi

  processed_lines=$((current_line - start_line + 1))
  percent=$(( 100 * current_line / total_lines ))

  # Time estimation
  now=$(date +%s)
  elapsed=$((now - start_time))
  if (( processed_lines > 1 )); then
    avg_time_per_line=$(awk "BEGIN {print $elapsed / $processed_lines}")
    remaining_lines=$((total_lines - current_line))
    remaining_time=$(awk "BEGIN {print $remaining_lines * $avg_time_per_line}")
    remaining_minutes=$(awk "BEGIN {print int(($remaining_time+59)/60)}")
  else
    remaining_minutes="?"
  fi

  echo "$bucket: $line $current_line/$total_lines ($percent%) | ETA: ${remaining_minutes} min"
  aws s3api restore-object \
    --restore-request "$restore_request" \
    --bucket "$bucket" \
    --profile "$profile" \
    --key "$line"
done < "$input_file"
