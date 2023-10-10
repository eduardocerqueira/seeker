#date: 2023-10-10T16:57:20Z
#url: https://api.github.com/gists/1d436a50ec44d999650a2f9e71cbef5c
#owner: https://api.github.com/users/kksudo

#!/usr/bin/env bash
set -eo pipefail

# Description: Test connection to AWS S3 bucket.
# Prerequisites:
# - Install jq https://stedolan.github.io/jq/download/
# - Install AWS CLI https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-install.html

# Author: Kirill https://Kazakov.xyz

# Usage:
# ./aws-s3-test-connection.sh <s3-bucket-name> <aws-access-key-id> <aws-secret-access-key> <aws-region>
# or
 "**********"# "**********"  "**********"D "**********"e "**********"f "**********"i "**********"n "**********"e "**********"  "**********"v "**********"a "**********"r "**********"i "**********"a "**********"b "**********"l "**********"e "**********"s "**********"  "**********"i "**********"n "**********"  "**********"y "**********"o "**********"u "**********"r "**********"  "**********"l "**********"o "**********"c "**********"a "**********"l "**********"  "**********"e "**********"n "**********"v "**********", "**********"  "**********"A "**********"W "**********"S "**********"_ "**********"A "**********"C "**********"C "**********"E "**********"S "**********"S "**********"_ "**********"K "**********"E "**********"Y "**********"_ "**********"I "**********"D "**********", "**********"  "**********"A "**********"W "**********"S "**********"_ "**********"A "**********"C "**********"C "**********"E "**********"S "**********"S "**********"_ "**********"K "**********"E "**********"Y "**********"_ "**********"I "**********"D "**********", "**********"  "**********"A "**********"W "**********"S "**********"_ "**********"R "**********"E "**********"G "**********"I "**********"O "**********"N "**********"  "**********"a "**********"n "**********"d "**********"  "**********"r "**********"u "**********"n "**********": "**********"
# ./aws-s3-test-connection.sh <s3-bucket-name>
# or
# Define variables in the .env.json file and run:
# ./aws-s3-test-connection.sh
# or
# Run script and enter variables in the interactive mode:
# ./aws-s3-test-connection.sh

# read variables from .env.json file
if [ -f .env.json ]; then
  echo "Reading variables from .env.json file..."
  export S3=$(jq -r '.BUCKET' .env.json)
  printf "S3: %s\n" "$S3"
  export AWS_ACCESS_KEY_ID= "**********"
  printf "AWS_ACCESS_KEY_ID: "**********"
  export AWS_SECRET_ACCESS_KEY= "**********"
  printf "AWS_SECRET_ACCESS_KEY: "**********"
  export AWS_REGION=$(jq -r '.REGION' .env.json)
  printf "AWS_REGION: %s\n" "$AWS_REGION"
fi

S3="${1:-$S3}"
# Check if S3 variable is empty or not equal to null
if [ -z "$S3" ] || [ "$S3" == "null" ]; then
  echo "Enter s3 bucket name:"
  read -r S3
fi
AWS_ACCESS_KEY_ID="${2: "**********"
 "**********"i "**********"f "**********"  "**********"[ "**********"  "**********"- "**********"z "**********"  "**********"" "**********"$ "**********"A "**********"W "**********"S "**********"_ "**********"A "**********"C "**********"C "**********"E "**********"S "**********"S "**********"_ "**********"K "**********"E "**********"Y "**********"_ "**********"I "**********"D "**********"" "**********"  "**********"] "**********"  "**********"| "**********"| "**********"  "**********"[ "**********"  "**********"" "**********"$ "**********"A "**********"W "**********"S "**********"_ "**********"A "**********"C "**********"C "**********"E "**********"S "**********"S "**********"_ "**********"K "**********"E "**********"Y "**********"_ "**********"I "**********"D "**********"" "**********"  "**********"= "**********"= "**********"  "**********"" "**********"n "**********"u "**********"l "**********"l "**********"" "**********"  "**********"] "**********"; "**********"  "**********"t "**********"h "**********"e "**********"n "**********"
  echo  "**********"Enter AWS_ACCESS_KEY_ID: "**********"
  read -r AWS_ACCESS_KEY_ID
fi
AWS_SECRET_ACCESS_KEY="${3: "**********"
 "**********"i "**********"f "**********"  "**********"[ "**********"  "**********"- "**********"z "**********"  "**********"" "**********"$ "**********"A "**********"W "**********"S "**********"_ "**********"S "**********"E "**********"C "**********"R "**********"E "**********"T "**********"_ "**********"A "**********"C "**********"C "**********"E "**********"S "**********"S "**********"_ "**********"K "**********"E "**********"Y "**********"" "**********"  "**********"] "**********"  "**********"| "**********"| "**********"  "**********"[ "**********"  "**********"" "**********"$ "**********"A "**********"W "**********"S "**********"_ "**********"S "**********"E "**********"C "**********"R "**********"E "**********"T "**********"_ "**********"A "**********"C "**********"C "**********"E "**********"S "**********"S "**********"_ "**********"K "**********"E "**********"Y "**********"" "**********"  "**********"= "**********"= "**********"  "**********"" "**********"n "**********"u "**********"l "**********"l "**********"" "**********"  "**********"] "**********"; "**********"  "**********"t "**********"h "**********"e "**********"n "**********"
  echo  "**********"Enter AWS_SECRET_ACCESS_KEY: "**********"
  read -r AWS_SECRET_ACCESS_KEY
fi
AWS_REGION="${4:-$AWS_REGION}"
if [ -z "$AWS_REGION" ] || [ "$AWS_REGION" == "null" ]; then
  echo "Enter AWS_REGION:"
  read -r AWS_REGION
fi

demo_file='hello.txt'

printf "\n===============================\n"
# configure aws  and check connection
printf "Setup AWS CLI settings...\n"
aws configure set aws_access_key_id "${AWS_ACCESS_KEY_ID}"
aws configure set aws_secret_access_key "${AWS_SECRET_ACCESS_KEY}"
aws configure set default.region "${AWS_REGION}"

printf "Testing connection to %s ...\n" "$S3"
# Generate a demo file to upload to s3
printf "Generating demo file...\n"
echo "Hello World from $(whoami)" > "${demo_file}"
echo "Uptime:$(uptime), current date:$(date)" >> "${demo_file}"
ls -la "${demo_file}"
cat "${demo_file}"
printf "\033[0;32mSuccess. Created a demo file.\033[0m\n"
echo "==============================="
printf "Uploading file to s3...\n"
aws s3 cp "${demo_file}" "s3://${S3}/${demo_file}"
printf "\033[0;32mSuccess. Uploaded a demo file.\033[0m\n"
printf "\nList files on s3...\n"
aws s3 ls "s3://${S3}/${demo_file}"
printf "\033[0;32mSuccess. List files on s3.\033[0m\n"
printf "\nDelete file %s from s3...\n" "${demo_file}"
aws s3 rm "s3://$S3/${demo_file}"
printf "\033[0;32mSuccess. Deleted file %s from s3.\033[0m\n" "${demo_file}"
printf "\n===============================\n"
printf "Done.\n"
