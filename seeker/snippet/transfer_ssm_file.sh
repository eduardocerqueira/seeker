#date: 2025-08-14T17:08:06Z
#url: https://api.github.com/gists/df0f56b98e5141646f4a8366325f4291
#owner: https://api.github.com/users/snehesh-thalapaneni

# This script will explain how to transfer a file to EC2 using SSM ONLY!

# You will need to have permission to run SSM commands on the target machine and have sudo access as well

# Infos
INSTANCE_ID=i-1234567890
FILE_NAME=the_file.tar.gz

# Step 1: Run command on machine to install netcat and dump from port to filename
# < Start session
aws ssm start-session --target $INSTANCE_ID --document-name
# < (On target machine) :
cd && sudo yum install nc -y && sudo nc -l -p 1234 > the_file.tar.gz

# Step 2: On another shell, open a port-forwarding session from your machine to the target machine
aws ssm start-session --target $INSTANCE_ID --document-name AWS-StartPortForwardingSession --parameters '{"portNumber":["1234"],"localPortNumber":["1234"]}'

# Step 3: On yet another shell, cat the source file into the transfer port on localhost over the tunnel
nc -w 3 127.0.0.1 1234 < the_file.tar.gz

# Step 4: Once the command in step 3 finishes, close all of the other shell sessions. Your file should be on the target now.