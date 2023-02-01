#date: 2023-02-01T16:45:10Z
#url: https://api.github.com/gists/2abb645e5a8104a552ed61e04f1253b1
#owner: https://api.github.com/users/lazyrabbit65

# first we download the list of IP ranges from CloudFlare
wget https://www.cloudflare.com/ips-v4

# set the security group ID
SG_ID="sg-00000000000000"

# iterate over the IP ranges in the downloaded file
# and allow access to ports 80 and 443
while read p
do
    aws ec2 authorize-security-group-ingress --group-id $SG_ID --ip-permissions IpProtocol=tcp,FromPort=80,ToPort=80,IpRanges="[{CidrIp=$p,Description='Cloudflare'}]"
    aws ec2 authorize-security-group-ingress --group-id $SG_ID --ip-permissions IpProtocol=tcp,FromPort=443,ToPort=443,IpRanges="[{CidrIp=$p,Description='Cloudflare'}]"
done< ips-v4

rm ips-v4