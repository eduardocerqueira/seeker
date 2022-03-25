#date: 2022-03-25T16:48:32Z
#url: https://api.github.com/gists/07410d1ca8195992d62298b28e0d3817
#owner: https://api.github.com/users/sebnyberg

#!/usr/bin/env bash
#
# Archive a folder by uploading it to S3 and
# renive it locally
#
# Usage: 
# 
# ./s3archive.sh $dir $bucket
#
echo "Uploading to archive bucket..."
aws s3 cp $1 "s3://${2}/${1}"
echo "Removing local files..."
rm -rf $1
echo "Done!"