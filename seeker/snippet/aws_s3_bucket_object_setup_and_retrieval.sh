#date: 2024-08-14T18:13:30Z
#url: https://api.github.com/gists/e078984ee9953dca2618e39f4e01d9df
#owner: https://api.github.com/users/terrancedejesus

#!/bin/bash

# Disable AWS CLI pager
export AWS_PAGER=""

# Step 1: Define variables
bucket_name="test-vulnerable-access-bucket-$(date +%s)"
region="us-east-1"  # Update if in a different region
object_prefix="sensitive-file"
num_objects=5
num_access_attempts=10

# Step 2: Create the S3 bucket
echo "[+] Creating a new S3 bucket: $bucket_name"
if [ "$region" == "us-east-1" ]; then
  aws s3api create-bucket --bucket $bucket_name --region $region
else
  aws s3api create-bucket --bucket $bucket_name --region $region --create-bucket-configuration LocationConstraint=$region
fi

if [ $? -ne 0 ]; then
  echo "[-] Error: Could not create S3 bucket."
  exit 1
fi

echo "[+] S3 bucket '$bucket_name' created."

# Step 2b: Disable Block Public Access settings (if enabled)
echo "[+] Disabling S3 Block Public Access settings for the bucket."
aws s3api put-public-access-block --bucket $bucket_name --public-access-block-configuration BlockPublicAcls=false,IgnorePublicAcls=false,BlockPublicPolicy=false,RestrictPublicBuckets=false

if [ $? -ne 0 ]; then
  echo "[-] Error: Could not disable Block Public Access settings."
  exit 1
fi
echo "[+] Block Public Access settings disabled."

# Step 3: Configure bucket policy to allow public access
echo "[+] Configuring bucket policy to allow public access."
aws s3api put-bucket-policy --bucket $bucket_name --policy '{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Principal": "*",
            "Action": "s3:GetObject",
            "Resource": "arn:aws:s3:::'"$bucket_name"'/*"
        }
    ]
}'

if [ $? -ne 0 ]; then
  echo "[-] Error: Could not set bucket policy."
  exit 1
fi

echo "[+] Bucket policy configured for public access."

# Step 4: Upload fake sensitive objects to the bucket
echo "[+] Uploading fake sensitive objects."
for i in $(seq 1 $num_objects); do
  echo "This is sensitive content $i" > "${object_prefix}-${i}.txt"
  aws s3 cp "${object_prefix}-${i}.txt" "s3://$bucket_name/"
  if [ $? -ne 0 ]; then
    echo "[-] Error: Could not upload object ${object_prefix}-${i}.txt."
    exit 1
  fi
  echo "[+] Uploaded ${object_prefix}-${i}.txt"
done

# Step 5: Access the objects publicly in rapid succession
echo "[+] Publicly accessing objects in rapid succession."
for i in $(seq 1 $num_access_attempts); do
  for j in $(seq 1 $num_objects); do
    curl -s "https://${bucket_name}.s3.${region}.amazonaws.com/${object_prefix}-${j}.txt" > /dev/null
    echo "[+] Accessed ${object_prefix}-${j}.txt (Attempt $i)"
  done
done

# Step 6: Cleanup - Remove objects and the S3 bucket
echo "[+] Cleaning up - Removing objects and the S3 bucket."

# Disable bucket versioning to allow deletion
aws s3api delete-bucket-policy --bucket $bucket_name

# Delete the objects
for i in $(seq 1 $num_objects); do
  aws s3 rm "s3://$bucket_name/${object_prefix}-${i}.txt"
  if [ $? -ne 0 ]; then
    echo "[-] Error: Could not delete object ${object_prefix}-${i}.txt."
    exit 1
  fi
  rm "${object_prefix}-${i}.txt"
  echo "[+] Deleted ${object_prefix}-${i}.txt"
done

# Delete the bucket
aws s3api delete-bucket --bucket $bucket_name --region $region

if [ $? -ne 0 ]; then
  echo "[-] Error: Could not delete S3 bucket."
  exit 1
fi

echo "[+] S3 bucket '$bucket_name' and objects deleted. Cleanup complete."
