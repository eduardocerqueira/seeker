#date: 2023-11-10T16:38:28Z
#url: https://api.github.com/gists/76623bbb8d58fe3988236a02065ce508
#owner: https://api.github.com/users/hugomoraismendes

aws --endpoint-url=http://localhost:4575 sns list-topics
# OR aws --endpoint-url=http://localstack:4575 sns list-topics
# {
#    "Topics": []
# }

aws --endpoint-url=http://localhost:4575 sns create-topic --name test-topic
# OR aws --endpoint-url=http://localstack:4575 sns create-topic --name test-topic
# {
#    "TopicArn": "arn:aws:sns:us-east-1:123456789012:test-topic"
# }

aws --endpoint-url=http://localhost:4575 sns list-topics
# OR aws --endpoint-url=http://localstack:4575 sns list-topics
# {
#    "Topics": [
#        {
#            "TopicArn": "arn:aws:sns:us-east-1:123456789012:test-topic"
#        }
#    ]
# }