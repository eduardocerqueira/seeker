#date: 2021-10-06T16:58:02Z
#url: https://api.github.com/gists/1115313f7c294bee757cf96a6dc4f7b9
#owner: https://api.github.com/users/minusworld

import json
import boto3
import os
import io

BUCKET="YOUR_BUCKET_HERE"

s3 = boto3.client("s3")

def lambda_handler(event, context):
    path = event['queryStringParameters'].get("path", "")
    item = event['queryStringParameters'].get("item", "latest")
    
    r = s3.list_objects_v2(
        Bucket=BUCKET,
        Prefix=path,
    )
    
    items = r['Contents']
    obj_key = f"{path}{'/' if path else ''}{item}"
    
    print(f"Getting '{obj_key}' from '{BUCKET}'")
    
    if item == "latest":
        match = list(sorted(items, key=lambda k: k['LastModified']))[-1]
    else:
        match = filter(lambda k: obj_key == k['Key'])
    
    buffer = io.BytesIO()
    s3.download_fileobj(BUCKET, match['Key'], buffer)
    buffer.seek(0)
    data = buffer.read().decode('utf-8')
    
    print(f"Got {len(data)} bytes: {data[:24]}...")
    
    return json.loads(data)
