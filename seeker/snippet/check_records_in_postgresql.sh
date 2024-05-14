#date: 2024-05-14T17:01:11Z
#url: https://api.github.com/gists/9037e3dd89fcdec9015055910e8e3ef9
#owner: https://api.github.com/users/dincosman

[postgres@ferret01 data]$ psql -d ferretdb
psql (16.2)
Type "help" for help.

-- For a mongodb database - a schema created in postgresql
-- For a mongodb collection - a relation/table created in postgresql
ferretdb=# SET schema 'test';
SET
ferretdb=# \dt
                    List of relations
 Schema |            Name             | Type  |  Owner   
--------+-----------------------------+-------+----------
 test   | _ferretdb_database_metadata | table | postgres
 test   | posts_4c2edfdc              | table | postgres
(2 rows)

ferretdb=# select count(*) from  posts_4c2edfdc;
  count  
---------
 1000000
(1 row)
