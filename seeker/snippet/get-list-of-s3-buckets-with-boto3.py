#date: 2022-12-06T16:57:57Z
#url: https://api.github.com/gists/4c0193aa57e41ec1e2b72adbf2b9f175
#owner: https://api.github.com/users/oscarojasgtz

import boto3

s3_resource = boto3.resource('s3')
buckets = []

for bucket in s3_resource.buckets.all():
    buckets.append(bucket.name)

print(buckets)
