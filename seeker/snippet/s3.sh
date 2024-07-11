#date: 2024-07-11T16:42:35Z
#url: https://api.github.com/gists/076a452faa6cbec0ab6f04a3e504f16f
#owner: https://api.github.com/users/Babatunde13

#!bin/bash

# This script will create a bucket in S3 and upload a file to it
aws s3 mb s3://bkoiki950assets

echo "Bucket created"

echo "THis is my first file in index" >> index.txt
echo "THis is another file in index1" >> index1.txt
echo "THis is another file in index2" >> index2.txt

aws s3 cp index.txt s3://bkoiki950assets
aws s3 cp index1.txt s3://bkoiki950assets
aws s3 cp index2.txt s3://bkoiki950assets

aws s3 sync . s3://bkoiki950assets # This will sync all files in the current directory to the bucket

echo "Files uploaded"

aws s3 ls s3://bkoiki950assets
