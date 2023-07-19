#date: 2023-07-19T17:00:41Z
#url: https://api.github.com/gists/e71d7ee1257336dff4ed875a2c73b976
#owner: https://api.github.com/users/jamiew

#!/bin/bash
# import helium denylist data from github to a local postgres table

url="https://raw.githubusercontent.com/helium/denylist/main/denylist.csv"
csv="/tmp/denylist.csv"
db="etl"

echo "downloading from $url ..."
curl -s "$url" > "$csv"
echo "saved to $csv. file has $(cat $csv | wc -l) rows"

# need to drop table first since postgres COPY FROM can't ignore duplicates
echo 'DROP TABLE "denylist"' | psql $db
echo 'CREATE TABLE "denylist" (
    "address" text NOT NULL,
    PRIMARY KEY ("address")
);' | psql $db

echo "COPY denylist (address) FROM '/tmp/denylist.csv';" | psql $db

echo "denylist table count:"
echo 'select count(*) from denylist' | psql $db