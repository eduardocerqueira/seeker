#date: 2023-11-10T16:38:28Z
#url: https://api.github.com/gists/76623bbb8d58fe3988236a02065ce508
#owner: https://api.github.com/users/hugomoraismendes

# (use any random email)
aws --endpoint-url=http://localhost:4575 sns subscribe --topic-arn arn:aws:sns:us-east-1:123456789012:test-topic --protocol email --notification-endpoint hugomorais@outlook.com
# OR aws --endpoint-url=http://localstack:4575 sns subscribe --topic-arn arn:aws:sns:us-east-1:123456789012:test-topic --protocol email --notification-endpoint hugomorais@outlook.com
# {
#    "SubscriptionArn": "arn:aws:sns:us-east-1:123456789012:test-topic:5aacffbe-ccf7-40d5-be97-c55af7392935"
# }

aws --endpoint-url=http://localhost:4575 sns list-subscriptions
# OR aws --endpoint-url=http://localstack:4575 sns list-subscriptions
# {
#    "Subscriptions": [
#        {
#            "Owner": "",
#            "Endpoint": "pibehatin@1rentcar.top",
#            "Protocol": "email",
#            "TopicArn": "arn:aws:sns:us-east-1:123456789012:test-topic",
#            "SubscriptionArn": "arn:aws:sns:us-east-1:123456789012:test-topic:5aacffbe-ccf7-40d5-be97-c55af7392935"
#        }
#    ]
# }
