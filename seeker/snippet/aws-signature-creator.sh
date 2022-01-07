#date: 2022-01-07T16:56:51Z
#url: https://api.github.com/gists/cff0b455a23859d3557232a1474bbcd6
#owner: https://api.github.com/users/jamini87

#!/usr/bin/env bash
# 
# File:
#   aws-signature-creator.sh
# 
# Description:
#   A signature creator for AWS signature version 4
# 
# References:
#   https://czak.pl/2015/09/15/s3-rest-api-with-curl.html
# 

# This area is to be filled in with the relevant information
# LOCAL_PATH has no error checking, make sure it exists at the local machine
# Use a . if wanting to stay in the local directory
# On MacOS (and others?) the path cannot use a ~ (e.g. /Users/user/Desktop instead of ~/Desktop)

readonly AWS_ACCESS_KEY_ID='<your_access_key_id>'
readonly AWS_SECRET_ACCESS_KEY='<your_secret_access_key>'
readonly AWS_SERVICE='s3'
readonly AWS_REGION='ap-southeast-1'
readonly AWS_S3_BUCKET_NAME='<your_bucket_name>'
readonly HTTP_CANONICAL_REQUEST_URI='/<directory>/<filename>'
readonly HTTP_REQUEST_CONTENT_TYPE='text/plain'
readonly LOCAL_DESTINATION='<your local filename>'
readonly LOCAL_PATH='<your local path>'

readonly AWS_SERVICE_ENDPOINT_URL="\
${AWS_S3_BUCKET_NAME}.${AWS_SERVICE}-${AWS_REGION}.amazonaws.com"

# Create an SHA-256 hash in hexadecimal.
# Usage:
#   hash_sha256 <string>
function hash_sha256 {
  printf "${1}" | openssl dgst -sha256 | sed 's/^.* //'
}

# Create an SHA-256 hmac in hexadecimal.
# Usage:
#   hmac_sha256 <key> <data>
function hmac_sha256 {
  key="$1"
  data="$2"
  printf "${data}" | openssl dgst -sha256 -mac HMAC -macopt "${key}" | \
      sed 's/^.* //'
}

readonly CURRENT_DATE_DAY="$(date -u '+%Y%m%d')"
readonly CURRENT_DATE_TIME="$(date -u '+%H%M%S')"
readonly CURRENT_DATE_ISO8601="${CURRENT_DATE_DAY}T${CURRENT_DATE_TIME}Z"

readonly HTTP_REQUEST_METHOD='GET'
readonly HTTP_REQUEST_PAYLOAD=''
readonly HTTP_REQUEST_PAYLOAD_HASH="$(printf "${HTTP_REQUEST_PAYLOAD}" | \
    openssl dgst -sha256 | sed 's/^.* //')"
readonly HTTP_CANONICAL_REQUEST_QUERY_STRING=''

readonly HTTP_CANONICAL_REQUEST_HEADERS="\
content-type:${HTTP_REQUEST_CONTENT_TYPE}
host:${AWS_SERVICE_ENDPOINT_URL}
x-amz-content-sha256:${HTTP_REQUEST_PAYLOAD_HASH}
x-amz-date:${CURRENT_DATE_ISO8601}"
# Note: The signed headers must match the canonical request headers.
readonly HTTP_REQUEST_SIGNED_HEADERS="\
content-type;host;x-amz-content-sha256;x-amz-date"

readonly HTTP_CANONICAL_REQUEST="\
${HTTP_REQUEST_METHOD}
${HTTP_CANONICAL_REQUEST_URI}
${HTTP_CANONICAL_REQUEST_QUERY_STRING}
${HTTP_CANONICAL_REQUEST_HEADERS}\n
${HTTP_REQUEST_SIGNED_HEADERS}
${HTTP_REQUEST_PAYLOAD_HASH}"

# Create the signature.
# Usage:
#   create_signature
function create_signature {
  stringToSign="AWS4-HMAC-SHA256
${CURRENT_DATE_ISO8601}
${CURRENT_DATE_DAY}/${AWS_REGION}/${AWS_SERVICE}/aws4_request
$(hash_sha256 "${HTTP_CANONICAL_REQUEST}")"

  dateKey=$(hmac_sha256 key:"AWS4${AWS_SECRET_ACCESS_KEY}" \
      "${CURRENT_DATE_DAY}")
  regionKey=$(hmac_sha256 hexkey:"${dateKey}" "${AWS_REGION}")
  serviceKey=$(hmac_sha256 hexkey:"${regionKey}" "${AWS_SERVICE}")
  signingKey=$(hmac_sha256 hexkey:"${serviceKey}" "aws4_request")

  printf "${stringToSign}" | openssl dgst -sha256 -mac HMAC -macopt \
      hexkey:"${signingKey}" | awk '{print $2}'
}

readonly SIGNATURE="$(create_signature)"

readonly HTTP_REQUEST_AUTHORIZATION_HEADER="\
AWS4-HMAC-SHA256 Credential=${AWS_ACCESS_KEY_ID}/${CURRENT_DATE_DAY}/\
${AWS_REGION}/${AWS_SERVICE}/aws4_request, \
SignedHeaders=${HTTP_REQUEST_SIGNED_HEADERS};x-amz-date, Signature=${SIGNATURE}"

cd "${LOCAL_PATH}"

curl -X "${HTTP_REQUEST_METHOD}" -v \
    "https://${AWS_SERVICE_ENDPOINT_URL}${HTTP_CANONICAL_REQUEST_URI}" \
    -H "Authorization: ${HTTP_REQUEST_AUTHORIZATION_HEADER}" \
    -H "content-type: ${HTTP_REQUEST_CONTENT_TYPE}" \
    -H "x-amz-content-sha256: ${HTTP_REQUEST_PAYLOAD_HASH}" \
    -H "x-amz-date: ${CURRENT_DATE_ISO8601}" \
    -o "${LOCAL_DESTINATION}"
    
cd -