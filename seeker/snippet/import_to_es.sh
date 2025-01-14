#date: 2025-01-14T16:41:54Z
#url: https://api.github.com/gists/e892fd20319dbd211d8f026c54ab0435
#owner: https://api.github.com/users/carsonip

# alternative to elasticsearch_loader
# not optimized, as http connections are not reused
# but simple enough to work for files that are not prohibitively large
pv foo.log | split --line-bytes=10000000 --filter="cat | sed 's/^/{\"create\":{\"_index\":\"logs-foo-default\"}}\n/' | curl -u admin:changeme -XPOST -H 'Content-Type: application/x-ndjson' 'http://localhost:9200/_bulk' --data-binary @-"