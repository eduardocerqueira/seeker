#date: 2025-04-16T17:09:59Z
#url: https://api.github.com/gists/d02d46b88b23e6b60cd2193bc09382d4
#owner: https://api.github.com/users/zbalkan

#!/usr/bin/env bash

set -o errexit
set -o nounset
set -o pipefail

cd "$(dirname "$0")"

db_file="investigations.db"

duckdb "$db_file" -c "
SELECT view_name, sql
FROM duckdb_views()
WHERE internal = false AND temporary = false;
"
