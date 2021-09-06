#date: 2021-09-06T17:11:39Z
#url: https://api.github.com/gists/5acde81b05caaddde909645db39a6329
#owner: https://api.github.com/users/tdk-rgare

# Get subnets and sort them by Availables IPs with th emost first

aws ec2 describe-subnets --query 'reverse(sort_by(Subnets, &AvailableIpAddressCount))[].[\
  AvailabilityZone, CidrBlock, SubnetId, SubnetArn, AvailableIpAddressCount, \
  Tags[?Key==`Name`][].Value]' \
  --color on --no-cli-pager
