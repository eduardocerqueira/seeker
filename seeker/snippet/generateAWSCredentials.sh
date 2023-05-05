#date: 2023-05-05T16:56:39Z
#url: https://api.github.com/gists/6e2966acce6d33578c7f74a2c1b91af2
#owner: https://api.github.com/users/imdurgadas

#!/bin/bash

set -e

# Set the profile name
profile_name=$1

# Run the command to export SSO credentials
sso_export=$(aws-sso-creds export --profile "$profile_name")

# Extract the access key, secret key, and session token from the export output
aws_access_key_id= "**********"= -f2)
aws_secret_access_key= "**********"= -f2)
aws_session_token= "**********"= -f2)

# Create or update the AWS credentials file
creds_file=~/.aws/credentials
if [ -f "$creds_file" ]; then
    # Check if the profile already exists in the file
    if grep -q "\[$profile_name\]" "$creds_file"; then
        # Update the credentials for the profile
        sed -i "/\[$profile_name\]/!b;n;caws_access_key_id= "**********"
        sed -i "/\[$profile_name\]/!b;n;caws_secret_access_key= "**********"
        sed -i "/\[$profile_name\]/!b;n;caws_session_token= "**********"
    else
        # Append the credentials for the new profile
        echo -e "\n[$profile_name]" >> "$creds_file"
        echo "aws_access_key_id= "**********"
        echo "aws_secret_access_key= "**********"
        echo "aws_session_token= "**********"
        echo "region=ap-south-1" >> "$creds_file"
    fi
else
    # Create a new credentials file
    cat > "$creds_file" <<EOF
[$profile_name]
aws_access_key_id= "**********"
aws_secret_access_key= "**********"
aws_session_token= "**********"
region=ap-south-1
EOF
fi

echo "AWS credentials file updated with credentials for profile '$profile_name'"ntials file
    cat > "$creds_file" <<EOF
[$profile_name]
aws_access_key_id=$aws_access_key_id
aws_secret_access_key=$aws_secret_access_key
aws_session_token=$aws_session_token
region=ap-south-1
EOF
fi

echo "AWS credentials file updated with credentials for profile '$profile_name'"