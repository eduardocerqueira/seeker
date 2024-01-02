#date: 2024-01-02T16:55:50Z
#url: https://api.github.com/gists/5e4a2cc1464fa8605c4a6eca03f21942
#owner: https://api.github.com/users/Youssef-Harby

import httpx
import datetime
import hashlib
import hmac

# AWS credentials
access_key = "**********"
secret_key = "**********"
region = "eu-central-1"  # e.g. 'us-west-1'
bucket = "bucket-name"
key = "testing.json"  # path to your file in the bucket

# Current timestamp
now = datetime.datetime.utcnow()
amz_date = now.strftime("%Y%m%dT%H%M%SZ")
date_stamp = now.strftime("%Y%m%d")  # Date w/o time, used in credential scope


# Create a function to sign key
def sign(key, msg):
    return hmac.new(key, msg.encode("utf-8"), hashlib.sha256).digest()


# Get a signing key
def get_signature_key(key, date_stamp, region_name, service_name):
    k_date = sign(("AWS4" + key).encode("utf-8"), date_stamp)
    k_region = sign(k_date, region_name)
    k_service = sign(k_region, service_name)
    k_signing = sign(k_service, "aws4_request")
    return k_signing


# Create a canonical request
method = "GET"
service = "s3"
host = f"{bucket}.s3.amazonaws.com"
endpoint = f"https://{host}/{key}"
payload_hash = hashlib.sha256(("").encode("utf-8")).hexdigest()
canonical_uri = f"/{key}"
canonical_querystring = ""
canonical_headers = (
    f"host:{host}\nx-amz-content-sha256:{payload_hash}\nx-amz-date:{amz_date}\n"
)
signed_headers = "host;x-amz-content-sha256;x-amz-date"
canonical_request = f"{method}\n{canonical_uri}\n{canonical_querystring}\n{canonical_headers}\n{signed_headers}\n{payload_hash}"

# Create the string to sign
algorithm = "AWS4-HMAC-SHA256"
credential_scope = f"{date_stamp}/{region}/{service}/aws4_request"
string_to_sign = f'{algorithm}\n{amz_date}\n{credential_scope}\n{hashlib.sha256(canonical_request.encode("utf-8")).hexdigest()}'

# Calculate the signature
signing_key = "**********"
signature = hmac.new(
    signing_key, string_to_sign.encode("utf-8"), hashlib.sha256
).hexdigest()

# Add signing information to the request headers
authorization_header = "**********"={access_key}/{credential_scope}, SignedHeaders={signed_headers}, Signature={signature}"
headers = {
    "x-amz-date": amz_date,
    "x-amz-content-sha256": payload_hash,
    "Authorization": authorization_header,
}

# Send the request
response = httpx.get(endpoint, headers=headers)

# Check response
if response.status_code == 200:
    print("File downloaded successfully.")
    print(response.content)
else:
    print(f"Failed to download file: {response.status_code}")

# Access the content
file_content = response.content
ccess the content
file_content = response.content
