#date: 2025-05-23T17:09:16Z
#url: https://api.github.com/gists/454b9750a014931421f5db1fbb005653
#owner: https://api.github.com/users/TiloGit

## IBM Cloud Test upload curl style
# make sure your endpoint is align with your S3 bucket otherwise you get "The specified bucket does not exist"
# seems that IBM want AWS Sign V4
# curl feature --aws-sigv4 (requires newer curl like 8.x) consider using container if your local curl is not new enough

# docker run -ti curlimages/curl sh
date=`date +%Fat%s`
fileName="test-tilo-${date}.txt"
echo "test tilo here at $date" > $fileName
s3Bucket="my-icos-bucket-mon01"
s3AccessKey="d5zzzzzzzzzzzzzzzzcba05"
s3SecretKey= "**********"
s3Region="any"

curl -X PUT \
    --user "${s3AccessKey}": "**********"
    --aws-sigv4 "aws:amz:${s3Region}:s3" \
    --upload-file ${fileName} \
    https://s3.mon01.cloud-object-storage.appdomain.cloud/${s3Bucket}/${fileName}


## Google Cloud Storage Test upload curl style
# make sure your endpoint is align with your S3 bucket otherwise you get "The specified bucket does not exist"
# seems that IBM want AWS Sign V4
# curl feature --aws-sigv4 (requires newer curl like 8.x) consider using container if your local curl is not new enough

# docker run -ti curlimages/curl sh
date=`date +%Fat%s`
fileName="test-tilo-${date}.txt"
echo "test tilo here at $date" > $fileName
s3Bucket="my-bucket-gcp123"
s3AccessKey="GOOG1EMzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzB4FQ7W"
s3SecretKey= "**********"
s3Region="any"

curl  -X PUT \
    --user "${s3AccessKey}": "**********"
    --aws-sigv4 "aws:amz:${s3Region}:s3" \
    --upload-file ${fileName} \
    http://storage.googleapis.com/${s3Bucket}/${fileName}


e} \
    http://storage.googleapis.com/${s3Bucket}/${fileName}


