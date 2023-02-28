#date: 2023-02-28T17:01:14Z
#url: https://api.github.com/gists/7116f7edd63b18be88633441eda7b167
#owner: https://api.github.com/users/mhynes-shopify

#!/usr/bin/env bash

dataset_id=os

echo "dataset_id, table_name, column_name"

for table in $(bq ls --format=prettyjson $dataset_id |jq -r '.[].tableReference.tableId' 2>/dev/null); do
    bq show --schema --dataset_id="$dataset_id" --format='json' "$table" 2>/dev/null \
        | jq '.[].name' \
        | sed -e "s/^/$dataset_id, $table, /g"
done
