#date: 2023-02-13T17:04:52Z
#url: https://api.github.com/gists/966de68e6cdb33fffdd0f48f63d0711a
#owner: https://api.github.com/users/hanif-ali-prtsr

# COUNT_GT 2
curl 'https://api-ps-3104-issue-with-uploaded-files-in-reports-rule.prtsr.io/api/reports/aggregate/' \
  -H 'content-type: application/json' \
  -H 'cookie: sessionid=<your_session_id>' \
  --data-raw '
    {"aggregate":[
      {
        "$match": {
          "$and": [
            {
              "formData.type_wuploadfilesinput2_ab66c8e4cff94f448f14a89590d15d8b_20230213143250.2": {
                "$exists": true
                }
            }
          ]
        }
      },
      {"$group":{
        "_id":"$metaData.status",
        "count":{
            "$sum":1
          }
        }
      }
    ]}' \
  --compressed

# COUNT_GTE 2
curl 'https://api-ps-3104-issue-with-uploaded-files-in-reports-rule.prtsr.io/api/reports/aggregate/' \
  -H 'content-type: application/json' \
  -H 'cookie: sessionid=<your_session_id>' \
  --data-raw '
    {"aggregate":[
      {
        "$match": {
          "$and": [
            {
              "formData.type_wuploadfilesinput2_ab66c8e4cff94f448f14a89590d15d8b_20230213143250.1": {
                "$exists": true
                }
            }
          ]
        }
      },
      {"$group":{
        "_id":"$metaData.status",
        "count":{
            "$sum":1
          }
        }
      }
    ]}' \
  --compressed

# COUNT_LT 2
curl 'https://api-ps-3104-issue-with-uploaded-files-in-reports-rule.prtsr.io/api/reports/aggregate/' \
  -H 'content-type: application/json' \
  -H 'cookie: sessionid=<your_session_id>' \
  --data-raw '
    {"aggregate":[
      {
        "$match": {
          "$and": [
            {
              "formData.type_wuploadfilesinput2_ab66c8e4cff94f448f14a89590d15d8b_20230213143250.1": {
                "$exists": false
                }
            }
          ]
        }
      },
      {"$group":{
        "_id":"$metaData.status",
        "count":{
            "$sum":1
          }
        }
      }
    ]}' \
  --compressed

# COUNT_LTE 2
curl 'https://api-ps-3104-issue-with-uploaded-files-in-reports-rule.prtsr.io/api/reports/aggregate/' \
  -H 'content-type: application/json' \
  -H 'cookie: sessionid=<your_session_id>' \
  --data-raw '
    {"aggregate":[
      {
        "$match": {
          "$and": [
            {
              "formData.type_wuploadfilesinput2_ab66c8e4cff94f448f14a89590d15d8b_20230213143250.2": {
                "$exists": false
                }
            }
          ]
        }
      },
      {"$group":{
        "_id":"$metaData.status",
        "count":{
            "$sum":1
          }
        }
      }
    ]}' \
  --compressed