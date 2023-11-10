#date: 2023-11-10T16:38:28Z
#url: https://api.github.com/gists/76623bbb8d58fe3988236a02065ce508
#owner: https://api.github.com/users/hugomoraismendes

aws --endpoint-url=http://localhost:4576 sqs create-queue --queue-name test_queue
# OR aws --endpoint-url=http://localstack:4576 sqs create-queue --queue-name test_queue
# {
#     "QueueUrl": "http://localhost:4576/123456789012/test_queue"
# }

aws --endpoint-url=http://localhost:4576 sqs list-queues
# OR aws --endpoint-url=http://localstack:4576 sqs list-queues
# {
#     "QueueUrls": [
#         "http://localhost:4576/123456789012/test_queue"
#     ]
# }