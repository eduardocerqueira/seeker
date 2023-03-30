#date: 2023-03-30T16:48:14Z
#url: https://api.github.com/gists/2bd45f83ec8eb79ad7c9a733027cd5e8
#owner: https://api.github.com/users/vampyar

docker exec -i postgres_container  bash -c "pg_dump --dbname=postgresql: "**********":$PGPASSWORD@$PGHOST:$PGPORT/$PGDB" > /asbolute/direction/path/dump.sql