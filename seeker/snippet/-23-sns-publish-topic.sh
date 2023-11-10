#date: 2023-11-10T16:38:28Z
#url: https://api.github.com/gists/76623bbb8d58fe3988236a02065ce508
#owner: https://api.github.com/users/hugomoraismendes

aws --endpoint-url=http://localhost:4575 sns publish  --topic-arn arn:aws:sns:us-east-1:123456789012:test-topic --message 'Test Message!'
# OR aws --endpoint-url=http://localstack:4575 sns publish  --topic-arn arn:aws:sns:us-east-1:123456789012:test-topic --message 'Test Message!'
# {
#    "MessageId": "n/a"
# }