#date: 2026-01-05T17:04:05Z
#url: https://api.github.com/gists/db77f4875cc108c7cf66331014d99aa9
#owner: https://api.github.com/users/dr-edmond-merkle

#!/bin/bash

# Variable Setup
name=First_Last
echo $name
email=first.last@merkle.com
echo $email
year=$(date +%Y)
echo $year
quarter=q$(( ($(date +%-m)-1)/3+1 ))
echo $quarter

# RSA variable setup
key_type=core_rsa
# Key comment
comment="${name}_${year}_${quarter}_key ${email}"
echo $comment
# file path
file=~/.ssh/${year}_${quarter}_${key_type}

# Example:
# ssh-keygen -t rsa -b 4096 -C "First_Last_2024_Q1_Key first.last@merkle.com"
ssh-keygen -t rsa -b 4096 -C "$comment" -f $file

# RSA variable setup
key_type=sftp_key
# Key comment
comment="${name}_${year}_${quarter}_key ${email}"
# file path
file=~/.ssh/${year}_${quarter}_${key_type}

# Example:
# ssh-keygen -t ed25519 -C "2024_q1_sftp_key first.last@merkle.com "
ssh-keygen -t ed25519 -C "$comment" -f $file

# Github variable setup
key_type=github
# Key comment
comment="${name}_${year}_${quarter}_key ${email}"
# file path
file=~/.ssh/${year}_${quarter}_${key_type}

ssh-keygen -t ed25519 -C "$comment" -f $file
#gh ssh-key add "$file.pub" --type authentication
gh auth login