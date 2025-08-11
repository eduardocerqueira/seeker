#date: 2025-08-11T17:10:53Z
#url: https://api.github.com/gists/72fdf58a0ac67c57082bcd4ac9c4862e
#owner: https://api.github.com/users/pradip-spring

# default: current region only
bash export_lambdas.sh

# pick regions
REGIONS="us-east-1 us-west-2" OUTDIR=./lambda_export bash export_lambdas.sh

# scan all regions that support Lambda
ALL_REGIONS=1 OUTDIR=./lambda_export bash export_lambdas.sh

# include S3 bucket notifications + CloudWatch Logs metadata
INCLUDE_S3=1 INCLUDE_LOGS=1 ALL_REGIONS=1 bash export_lambdas.sh