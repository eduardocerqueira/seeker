#date: 2022-06-16T17:15:00Z
#url: https://api.github.com/gists/f0323ff2bca90951567fe3800fdb59c1
#owner: https://api.github.com/users/nulconaux

#!/bin/bash

DB_NAME=${1}
export PGPASSWORD=
DBMS_SHELL="psql -p 5432 -h localhost"
DBMS_USER="postgres"

#if [ "$1" = '--help' ]; then
if [[ ( "$1" == '--help' ) || ( "$1" == '-h' ) ]]; then
        echo "usage: $0 [DB_NAME] [DBMS_SHELL]"
        echo "default DB_NAME is your username"
        echo "default DBMS_SHELL is 'psql'"
        echo "default DBMS_USER is 'postgres'"
        exit 0
fi

if [ -n "$1" ]
then DB_NAME="$1"
fi
if [ -n "$2" ]
then DBMS_SHELL="$2"
fi
if [ -n "$3" ]
then DBMS_USER="$3"
fi

alias echo='>&2 echo'

mkdir -p "$DB_NAME"
echo "Fetching table list ..."
$DBMS_SHELL "$DB_NAME" -U $DBMS_USER -c "copy (select table_name from information_schema.tables where table_schema='public') to STDOUT;" > "$DB_NAME/tables.txt"
dbms_success=$?
if ! [ $dbms_success ]
then exit 4
fi

echo "Fetching tables ..."
readarray tables < "$DB_NAME/tables.txt"
for t in ${tables[*]}; do
        $DBMS_SHELL -d "$DB_NAME" -U $DBMS_USER -c "copy (select * from $t) to STDOUT with delimiter ',' CSV HEADER;" > "$DB_NAME/$t.csv"
done