#date: 2023-11-28T17:00:53Z
#url: https://api.github.com/gists/281b2edb8da1b141cb1c6e9663e3cecb
#owner: https://api.github.com/users/senojsitruc

#!/bin/bash

# List of hard-coded table names
tables=("table1", "table2")

# Iterate over the table names
for table in "${tables[@]}"; do

  # Output file name
  file_name="output-$table.sql"

  echo "$file_name"

  # Execute sqlite3 command remotely via SSH
  ssh bedrock@10.10.10.123 "sqlite3 Bedrock/bedrock.db \".dump $table\"" | \
    # Filter lines starting with "INSERT INTO"
    grep '^INSERT INTO' | \

    # Replace "INSERT INTO" with "INSERT OR UPDATE INTO"
    sed 's/INSERT INTO/INSERT OR REPLACE INTO/' >> "$file_name"

    # Use "npx wrangler" to send the data
    npx wrangler d1 execute penguin-prod -y --file="$file_name"
done