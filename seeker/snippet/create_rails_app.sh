#date: 2023-06-02T17:05:00Z
#url: https://api.github.com/gists/ead3374636792995ad6627a0cd88cb57
#owner: https://api.github.com/users/mjmor

#!/bin/sh

# Set local ruby version to be used to install rails. Use 3.0.6 to be compatible with AWS Elastic Beanstalk.
rbenv local 3.0.6

# Install a rails app in the current directory.
# Consider skipping unneeded gems. E.g. --skip-action-mailer --skip-action-mailbox 
# if we don't need to send / recv email. See rails new -h for full list of options.
rails new ./ --skip-action-mailer --skip-action-mailbox --skip-active-storage