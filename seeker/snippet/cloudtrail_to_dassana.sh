#date: 2022-05-05T16:56:26Z
#url: https://api.github.com/gists/158904fce6e0419f874dfb5d65b524dd
#owner: https://api.github.com/users/gauravphoenix

curl -H 'Content-type: application/x-ndjson' \
-H 'x-dassana-app-id: foo' \
-H 'x-dassana-token: TOKEN_GOES_HERE' \
-H'Content-Encoding: gzip' \
-H 'x-dassana-data-key: Records' \
https://ingestion.dassana.cloud/logs --data-binary @/PATH/TO/CLOUDTRAIL_FILE.GZ