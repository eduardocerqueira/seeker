#date: 2023-11-10T16:38:28Z
#url: https://api.github.com/gists/76623bbb8d58fe3988236a02065ce508
#owner: https://api.github.com/users/hugomoraismendes

aws --endpoint-url=http://localhost:4572 s3 rm s3://mytestbucket/mongo.log
# OR aws --endpoint-url=http://localstack:4572 s3 rm s3://mytestbucket/mongo.log
# delete: s3://mytestbucket/mongo.log

aws --endpoint-url=http://localhost:4572 s3 ls s3://mytestbucket
# OR aws --endpoint-url=http://localstack:4572 s3 ls s3://mytestbucket