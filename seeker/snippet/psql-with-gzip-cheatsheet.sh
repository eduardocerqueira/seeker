#date: 2025-01-07T16:50:35Z
#url: https://api.github.com/gists/0c6286c181adaa43dce39887538d09f3
#owner: https://api.github.com/users/shakahl

# This is just a cheat sheet:

# On production
sudo -u postgres pg_dump database | gzip -9 > database.sql.gz

# On local
scp -C production:~/database.sql.gz
dropdb database && createdb database
gunzip < database.sql.gz | psql database