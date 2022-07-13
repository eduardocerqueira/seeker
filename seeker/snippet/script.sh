#date: 2022-07-13T16:53:09Z
#url: https://api.github.com/gists/76af2e9169ed6c2cf8960074db65a106
#owner: https://api.github.com/users/droman93

# best practice: linux
nano ~/.pgpass
*:5432:*:username:password
chmod 0600 ~/.pgpass

# best practice: windows
edit %APPDATA%\postgresql\pgpass.conf
*:5432:*:username:password

# linux
PGPASSWORD="password" pg_dump --no-owner -h host -p port -U username database > file.sql

# windows
PGPASSWORD=password&& pg_dump --no-owner -h host -p port -U username database > file.sql

# alternative
pg_dump --no-owner --dbname=postgresql://username:password@host:port/database > file.sql

# restore
psql --set ON_ERROR_STOP=on -U postgres -d database -1 -f file.sql
pg_restore --no-privileges --no-owner -U postgres -d database --clean file.sql # only works for special dumps

# backup excluding table
pg_dump --no-owner -h 127.0.0.1 -p 5432 -U username --exclude-table=table1 --exclude-table=table2 --exclude-table=table1_id_seq --exclude-table=table2_id_seq database > tmp.sql

# backup including only table
pg_dump --no-owner -h 127.0.0.1 -p 5432 -U username --table=table1 --table=table2 database > tmp.sql

# backup only schema (no data)
pg_dump --no-owner -h 127.0.0.1 -p 5432 -U username --schema-only database > tmp.sql

# transfer database with 2 tables empty
pg_dump --no-owner -h 127.0.0.1 -p 5432 -U username --exclude-table=table1 --exclude-table=table2 --exclude-table=table1_id_seq --exclude-table=table2_id_seq database > 1.sql
pg_dump --no-owner -h 127.0.0.1 -p 5432 -U username --table=table1 --table=table2 --schema-only database > 2.sql
psql --set ON_ERROR_STOP=on -U username -d database -1 -f 1.sql
psql --set ON_ERROR_STOP=on -U username -d database -1 -f 2.sql

# backup and restore
PGPASSWORD=password && pg_dump --no-owner -h 127.0.0.1 -p 5432 -U username database > tmp.sql
psql -U postgres -d database -c "drop schema public cascade; create schema public;"
psql --set ON_ERROR_STOP=on -U postgres -d database -1 -f tmp.sql
rm tmp.sql

# if you need to remove unintentionally backuped owners afterwards
sed -i.bak '/OWNER TO specialowner/d' input.sql