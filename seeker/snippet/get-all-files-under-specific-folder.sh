#date: 2025-09-26T17:02:23Z
#url: https://api.github.com/gists/5f1ec0201634b8224c3f17cbf3fb9ac8
#owner: https://api.github.com/users/abrenner

curl --location --request GET 'https://adam.vssw.hop.delllabs.net:9200/isi-metadataiq-index.ioflash.04bf1be5f95e87b4fa6512225f9ec530ca70/_search' \
--header 'Content-Type: application/json' \
--header 'Authorization: Basic HIDDEN' \
--data '{
    "query": {
            "bool": {
                "must": [
                    {
                        "match_phrase_prefix": {
                            "data.path": "/ifs/midx/"
                        }
                    }
                ]
            }
    },
    "aggs": {
        "physical_space_used_in_bytes": {
            "sum": { 
                "field": "data.physical_size" 
            } 
        },
        "space_used_in_bytes": {
            "sum": { 
                "field": "data.size" 
            } 
        }
    }
}'