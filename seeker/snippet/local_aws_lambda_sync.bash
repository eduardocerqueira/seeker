#date: 2023-11-06T16:53:29Z
#url: https://api.github.com/gists/00952b9e941934ce191436f6b53a14f5
#owner: https://api.github.com/users/purvesh62

#!/bin/bash
LAMBDA_NAME=$1
ZIP_FILE=$2
if [[ -z $LAMBDA_NAME || -z $ZIP_FILE ]]; then
    echo "Usage: $0 <lambda_function_name> <zip_file_name>"
    exit 1
fi
# Construct full path
FILE_PATH="${PWD}/${ZIP_FILE}"
# Adding "fileb://" prefix
FILEB_PATH="fileb://${FILE_PATH}"
while true; do
    # Prompt the user to press a key to continue
    echo "Press any key to prepare for upload..."
    read -n 1 -s
    # Zip the file
    zip -q -r $ZIP_FILE . -x "$ZIP_FILE" ".git/*"
    echo "About to upload the following file:"
    echo "File Path: $FILE_PATH"
    echo "Lambda Function: $LAMBDA_NAME"
    echo "Do you wish to proceed with the upload? (y/n)"
    read -n 1 -s confirm
    if [ "$confirm" == "y" ] || [ "$confirm" == "Y" ]; then
        # Upload the file to Lambda
        aws lambda update-function-code --function-name $LAMBDA_NAME --zip-file $FILEB_PATH
        echo "Upload done..."
    else
        echo "Upload cancelled."
    fi
done