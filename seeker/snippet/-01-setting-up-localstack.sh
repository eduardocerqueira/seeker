#date: 2023-11-10T16:38:28Z
#url: https://api.github.com/gists/76623bbb8d58fe3988236a02065ce508
#owner: https://api.github.com/users/hugomoraismendes

git clone https://github.com/localstack/localstack.git
cd localstack
make clean install test
make infra
# . .venv/bin/activate; exec localstack/mock/infra.py
# Starting local dev environment. CTRL-C to quit.
# Starting local Elasticsearch (port 4571)...
# Starting mock ES service (port 4578)...
# Starting mock S3 server (port 4572)...
# Starting mock SNS server (port 4575)...
# Starting mock SQS server (port 4576)...
# Starting mock API Gateway (port 4567)...
# Starting mock DynamoDB (port 4569)...
# Starting mock DynamoDB Streams (port 4570)...
# Starting mock Firehose (port 4573)...
# Starting mock Lambda (port 4574)...
# Starting mock Kinesis (port 4568)...
Starting mock Redshift server (port 4577)...
Ready.