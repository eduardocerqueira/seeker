#date: 2023-05-30T16:59:59Z
#url: https://api.github.com/gists/04cecef1df41c7ceed249f30fc607bfc
#owner: https://api.github.com/users/joshualouden

# You don't need Fog in Ruby or some other library to upload to S3 -- shell works perfectly fine
# This is how I upload my new Sol Trader builds (http://soltrader.net)
# Based on a modified script from here: http://tmont.com/blargh/2014/1/uploading-to-s3-in-bash

# ====================================================================================
# Aug 25, 2016 sh1n0b1
# Modified this script to support AWS session token
# More work will be done on this.
#
# S3KEY="ASIAJLFN####################"
# S3SECRET= "**********"
# S3SESSION="FQoDYXdzELT//////////###########################################"
# ====================================================================================


S3KEY=""
S3SECRET= "**********"
S3SESSION=""

# No need to modify the function below:
function putS3
{
  bucket=$1
  path=$2
  file=$3
  aws_path=$4
  date=$(date +"%a, %d %b %Y %T %z")
  acl="x-amz-acl:public-read"
  content_type='application/x-compressed-tar'
  token="x-amz-security-token: "**********"
  string= "**********"
  signature= "**********"
  
  # Executing
  # curl -v -X PUT -T "$path/$file" \
  #   -H "Host: $bucket.s3.amazonaws.com" \
  #   -H "Date: $date" \
  #   -H "Content-Type: $content_type" \
  #   -H "$acl" \
  #   -H "$token" \
  #   -H "Authorization: AWS ${S3KEY}:$signature" \
  #   "https://$bucket.s3.amazonaws.com$aws_path$file"

  # Print the curl command
  echo "curl -v -X PUT -T \"$path/$file\" \\"
  echo "  -H \"Host: $bucket.s3.amazonaws.com\" \\" 
  echo "  -H \"Date: $date\" \\"
  echo "  -H \"Content-Type: $content_type\" \\"
  echo "  -H \"$acl\" \\"
  echo "  -H \"$token\" \\"
  echo "  -H \"Authorization: AWS ${S3KEY}:$signature\" \\"
  echo "  \"https://$bucket.s3.amazonaws.com$aws_path$file\""

}
function getObject
{
  bucket=$1
  filepath=$2
  date=$(date +"%a, %d %b %Y %T %z")
  acl="x-amz-acl:public-read"
  content_type='application/x-compressed-tar'
  token="x-amz-security-token: "**********"
  string= "**********"
  signature= "**********"
  

  # #Executing
  # curl -v -X GET \
  #   -H "Host: $bucket.s3.amazonaws.com" \
  #   -H "Date: $date" \
  #   -H "Content-Type: $content_type" \
  #   -H "$acl" \
  #   -H "$token" \
  #   -H "Authorization: AWS ${S3KEY}:$signature" \
  #   "https://$bucket.s3.amazonaws.com$filepath"

  # Print the curl command
  echo "curl -v -X GET \\"
  echo "  -H \"Host: $bucket.s3.amazonaws.com\" \\" 
  echo "  -H \"Date: $date\" \\"
  echo "  -H \"Content-Type: $content_type\" \\"
  echo "  -H \"$acl\" \\"
  echo "  -H \"$token\" \\"
  echo "  -H \"Authorization: AWS ${S3KEY}:$signature\" \\"
  echo "  \"https://$bucket.s3.amazonaws.com$filepath\""
}
function listBucket
{
  bucket=$1
  filepath=$2
  date=$(date +"%a, %d %b %Y %T %z")
  acl="x-amz-acl:public-read"
  content_type='application/x-compressed-tar'
  token="x-amz-security-token: "**********"
  string= "**********"
  signature= "**********"
  

  # #Executing
  # curl -v -X GET \
  #   -H "Host: $bucket.s3.amazonaws.com" \
  #   -H "Date: $date" \
  #   -H "Content-Type: $content_type" \
  #   -H "$acl" \
  #   -H "$token" \
  #   -H "Authorization: AWS ${S3KEY}:$signature" \
  #   "https://$bucket.s3.amazonaws.com$filepath?list-type=2"

  # Print the curl command
  echo "curl -v -X GET \\"
  echo "  -H \"Host: $bucket.s3.amazonaws.com\" \\" 
  echo "  -H \"Date: $date\" \\"
  echo "  -H \"Content-Type: $content_type\" \\"
  echo "  -H \"$acl\" \\"
  echo "  -H \"$token\" \\"
  echo "  -H \"Authorization: AWS ${S3KEY}:$signature\" \\"
  echo "  \"https://$bucket.s3.amazonaws.com$filepath?list-type=2\""
}
# ====================================================================================
# putS3 "bucket_name" "/local_filepath" "upload_file.txt" "/s3_dir"
# Just replace the parameters above^ 
# ====================================================================================
# ====================================================================================
# getObject "bucket_name" "/file_path.txt"
# ====================================================================================
# ====================================================================================
#listBucket "bucket_name" "/dir"
# ====================================================================================
============
# getObject "bucket_name" "/file_path.txt"
# ====================================================================================
# ====================================================================================
#listBucket "bucket_name" "/dir"
# ====================================================================================
