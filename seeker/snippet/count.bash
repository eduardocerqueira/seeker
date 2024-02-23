#date: 2024-02-23T17:02:48Z
#url: https://api.github.com/gists/106824e57cee383eaa87dbc5e56a5a97
#owner: https://api.github.com/users/patrickyee23

# count only
aws s3 ls s3://ttam-data-xfer-clinic-fulgent-us-west-2/wes/raw_data/ | wc

# count micronic code
aws s3 ls s3://ttam-data-xfer-clinic-fulgent-us-west-2/wes/raw_data/ | tr -s \  | cut -d' ' -f4 | cut -d'-' -f 1 | sort | uniq -c

# count file types
aws s3 ls s3://ttam-data-xfer-clinic-fulgent-us-west-2/wes/raw_data/ | tr -s \  | cut -d' ' -f4 | cut -d'.' -f 2 | sort | uniq -c
  