#date: 2023-11-10T16:38:28Z
#url: https://api.github.com/gists/76623bbb8d58fe3988236a02065ce508
#owner: https://api.github.com/users/hugomoraismendes

aws --endpoint-url=http://localhost:4572 s3 cp /tmp/mongo.log s3://mytestbucket
# OR aws --endpoint-url=http://localstack:4572 s3 cp /tmp/mongo.log s3://mytestbucket
# upload: ../../../../tmp/mongo.log to s3://mytestbucket/mongo.log

aws --endpoint-url=http://localhost:4572 s3 ls s3://mytestbucket
# OR aws --endpoint-url=http://localstack:4572 s3 ls s3://mytestbucket
# 2017-04-05 01:18:39       4789 mongo.log