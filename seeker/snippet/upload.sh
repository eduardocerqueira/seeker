#date: 2021-11-04T17:14:24Z
#url: https://api.github.com/gists/6e1b3984ae959c7b9ce90e62aa5fb8b0
#owner: https://api.github.com/users/hawyar

#!/bin/bash

source="$1"
target="$2"
profile="$3"

# max_concurrent_requests=15 # default 10
# multipart_threshold=100MB
# multipart_chunksize=50MB


if [ -z "$source" ]; then
  echo "Usage: upload.sh <source> <profile>"
  exit 1
fi

if ! [ -x "$(command -v aws)" ]; then
  echo 'Error: aws cli is not installed.' >&2

  # curl "https://awscli.amazonaws.com/AWSCLIV2.pkg" -o "AWSCLIV2.pkg"
  # sudo installer -pkg AWSCLIV2.pkg -target /
  # rm AWSCLIV2.pkg
  
  if ! [ -x "$(command -v aws)" ]; then
	echo 'Error: aws cli is still not installed.' >&2
	exit 1
  fi
fi

if [ -z "$target" ]; then
  target="toS3"
fi

files=`ls ${source}`

for file in $files
do
	# csv and json 
	if [[ $file == *.csv ]] || [[ $file == *.json ]]
	then
	echo "uploading $source/$file to $target"
	
	# aws s3 cp $source/$file s3://$profile/$target/$file	--max_concurrent_requests=$max_concurrent_requests
	# --multipart-threshold=$multipart_threshold
	# --multipart-chunksize=$multipart_chunksize
	
	fi
	else
	echo "skipping $source/$file"
	fi
done