#date: 2023-05-16T17:07:16Z
#url: https://api.github.com/gists/3c82fbb391f08ba0ba2d82a129ba98f4
#owner: https://api.github.com/users/benmotyka

aws ec2 describe-instances --filters "Name=instance-state-name,Values=running" --query 'Reservations[*].Instances[*].[Tags[?Key==`Name`].Value,PrivateIpAddress,KeyName,`--------------------`]'  --output table
