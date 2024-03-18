#date: 2024-03-18T17:10:21Z
#url: https://api.github.com/gists/013c036be1e67c43629afb144554bc62
#owner: https://api.github.com/users/Akintola

#!/bin/bash

PATH=/usr/local/sbin:/usr/local/bin:/sbin:/bin:/usr/sbin:/usr/bin
export DISPLAY=:0.0

# AWS S3 Bucket Details
AWS_BUCKET=""
AWS_REGION=""

# Export AWS credentials or use AWS profile
export AWS_ACCESS_KEY_ID= "**********"
export AWS_SECRET_ACCESS_KEY= "**********"
export AWS_DEFAULT_REGION="$AWS_REGION"

# Local Folder Details
LOCAL_FOLDER=""

# Azure Blob Container Details
AZURE_STORAGE_ACCOUNT=""
AZURE_CONTAINER=""
AZURE_STORAGE_KEY=""

# Sync from AWS S3 to local folder
echo "Syncing from AWS S3 to local folder..."
aws s3 sync s3://$AWS_BUCKET $LOCAL_FOLDER 

# Sync from local folder to Azure Blob Container using AzCopy
echo "Syncing from local folder to Azure Blob Container..."
azcopy sync "$LOCAL_FOLDER" "https://$AZURE_STORAGE_ACCOUNT.blob.core.windows.net/$AZURE_CONTAINER?$AZURE_STORAGE_KEY" --recursive

echo "Sync completed successfully."