#date: 2024-09-09T16:51:17Z
#url: https://api.github.com/gists/acf85718e9a5443c155ca7b4e1d79f35
#owner: https://api.github.com/users/oswaldom-code

# This secure deletion script (.sh) is designed to securely delete the .env file in Linux development 
# environments. The .env file typically contains sensitive environment variables such as database 
# credentials, authentication tokens, etc. Before deleting the .env file, the script overwrites its 
# content with random data using OpenSSL to prevent the original content from being recovered by 
# unauthorized users.
# 
# Usage Instructions:
# 1. Save this script in the same directory as the .env file you wish to delete.
# 2. Run the script by executing it from the Linux command line.
# 3. The script will check if the .env file exists.
# 4. If the .env file exists, it will overwrite its content with random data and then securely delete it.
# 5. If the .env file does not exist, the script will display a message indicating that the file was not 
# found.
# 
# Note: This script assumes that OpenSSL is installed on your Linux system. Ensure you have OpenSSL 
# installed for the script to function correctly.
#!/bin/bash

# Check if the .env file exists
if [ -f .env ]; then
    # Overwrite the contents of the file with random data
    echo "Overwriting .env file with random data..."
    openssl rand -out .env -base64 $(stat -c %s .env)
    
    # Delete the file
    echo "Deleting .env file..."
    rm -f .env
    
    echo ".env file has been securely deleted."
else
    echo ".env file does not exist."
fi
