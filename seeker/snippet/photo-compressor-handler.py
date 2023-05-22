#date: 2023-05-22T17:09:36Z
#url: https://api.github.com/gists/675ac17706d4860d6cce7afb460dc128
#owner: https://api.github.com/users/tomersmadja

import json
import urllib.parse
import boto3
import io
from PIL import Image 
# to use PIL, add the follwoing layer - arn:aws:lambda:eu-north-1:113088814899:layer:Klayers-python37-Pillow:11

print('Loading function')

s3 = boto3.client('s3')

def lambda_handler(event, context):
    # Get the object from the event and show its content type
    bucket = event['Records'][0]['s3']['bucket']['name']
    key = urllib.parse.unquote_plus(event['Records'][0]['s3']['object']['key'], encoding='utf-8')
    try:
        # Open the image
        file_byte_string = s3.get_object(Bucket=bucket, Key=key)['Body'].read()
        img = Image.open(io.BytesIO(file_byte_string))
        
        # Save the compressed image to an in-memory file
        in_mem_file = io.BytesIO()
        img.save(
            in_mem_file, 
            format=img.format,
            optimize = True,
            quality = 10
            )
        in_mem_file.seek(0)
        
        output_key = key.replace('raw', 'comprassed')
        print(f'going to save file to {bucket}/{output_key}')
        
        # Upload image to s3
        s3.upload_fileobj(
            in_mem_file,
            bucket,
            output_key
        )
    except Exception as e:
        print(e)
        print('Error getting object {} from bucket {}. Make sure they exist and your bucket is in the same region as this function.'.format(key, bucket))
        raise e
        
print('Function was successfully done')
