#date: 2024-05-14T16:53:14Z
#url: https://api.github.com/gists/925bb6757d5d5579c1e06681df6549fd
#owner: https://api.github.com/users/dincosman

osmandinc@192 mongoapi % node 1mri_ferret.js

(node:25327) [DEP0040] DeprecationWarning: The `punycode` module is deprecated. Please use a userland alternative instead.
(Use `node --trace-deprecation ...` to show where the warning was created)
Inserted: 1000000 rows
Operation took: 65403.59325 milliseconds

[root@mongo01 ~]# mongosh mongodb://postgres:postgres@192.168.60.202:27002/ferretdb?authMechanism=PLAIN
Current Mongosh Log ID: 6636743d70a76b19852202d7
Connecting to:          mongodb://<credentials>@192.168.60.202:27002/ferretdb?authMechanism=PLAIN&directConnection=true&appName=mongosh+2.2.5
Using MongoDB:          7.0.42
Using Mongosh:          2.2.5

For mongosh info see: https://docs.mongodb.com/mongodb-shell/

------
   The server generated these startup warnings when booting
   2024-05-04T17:45:33.177Z: Powered by FerretDB v1.21.0 and PostgreSQL 16.2 on x86_64-pc-linux-gnu, compiled by gcc.
   2024-05-04T17:45:33.177Z: Please star us on GitHub: https://github.com/FerretDB/FerretDB.
   2024-05-04T17:45:33.177Z: The telemetry state is undecided.
   2024-05-04T17:45:33.177Z: Read more about FerretDB telemetry and how to opt out at https://beacon.ferretdb.com.
------
ferretdb> use test;
switched to db test
test> db.posts.count();
1000000
