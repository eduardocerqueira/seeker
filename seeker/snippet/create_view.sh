#date: 2025-04-16T17:09:26Z
#url: https://api.github.com/gists/b6b6af0981698ae8c314a384aa3cbed3
#owner: https://api.github.com/users/zbalkan

#!/usr/bin/env bash

set -o errexit
set -o nounset
set -o pipefail

if [[ "${TRACE-0}" == "1" ]]; then
    set -o xtrace
fi

if [[ "${1-}" =~ ^-*h(elp)?$ || $# -lt 2 ]]; then
    echo "Usage: $0 <case-name> <date-pattern>"
    echo "Example: $0 SI-801 2025-04-0*"
    exit 1
fi

cd "$(dirname "$0")"

case_name="$1"
date_pattern="$2"

# Normalize case name to lowercase, snake_case
view_name="$(echo "$case_name" | tr '[:upper:]' '[:lower:]' | tr '-' '_' | tr ' ' '_')"

# Database file
db_file="investigations.db"

# Resolve file paths
file_paths=$(ls /NFS/${date_pattern}-siem*.log.gz 2>/dev/null | sed 's/^/"/;s/$/"/' | paste -sd, -)

if [[ -z "$file_paths" ]]; then
    echo "No files matched pattern: /NFS/${date_pattern}-siem*.log.gz"
    exit 1
fi

# Create the view in DuckDB
duckdb "$db_file" <<EOF
CREATE OR REPLACE VIEW $view_name AS
SELECT
    CAST(timestamp AS TIMESTAMP) AS timestamp,
    agent.id AS agent_id,
    agent.name AS agent_name,
    location,
    decoder.name AS decoder,
    full_log
FROM read_ndjson_objects_auto([${file_paths}]);
EOF

echo "View '$view_name' created in $db_file"
