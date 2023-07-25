#date: 2023-07-25T17:08:24Z
#url: https://api.github.com/gists/f4b7f83077d6af83a795e3c864d37a93
#owner: https://api.github.com/users/chicagobuss

#!/usr/bin/env python

import boto3

from boto3.dynamodb.conditions import Key, Attr
from datetime import datetime

def get_hourly_usage(table, run_ts):

    now = datetime.fromtimestamp(run_ts)
    # TODO: Consider datetime edge cases
    day_str = now.strftime("%Y%m%d")
    hour_start_ts = run_ts - run_ts % 3600
    hour_end_ts = hour_start_ts + 3600
    fivemin_start_ts = run_ts - run_ts % 300
    fivemin_end_ts = fivemin_start_ts + 300
    print("getting five-minute usage aggregation with run_ts %s for day %s between %s and %s" % \
         (run_ts, day_str, fivemin_start_ts, fivemin_end_ts))
    
    response = table.query(
        IndexName="timestamp-index",
        KeyConditionExpression = Key( 
           'day').eq(day_str) & Key( \
           'timestamp').between(hour_start_ts,hour_end_ts),
        FilterExpression="attribute_exists(#cust_id)",
        # this is how you can use hyphens in column names
        ExpressionAttributeNames={"#cust_id": "cust-id"}
    )

    for item in response["Items"]:
        print(item)


if __name__ == '__main__':
    
    boto3.setup_default_session(profile_name='dev')

    TABLE_NAME = 'usage'

    # ddb Table resource
    dynamodb = boto3.resource('dynamodb', region_name='us-east-1')
    table = dynamodb.Table(TABLE_NAME)
    run_ts = 1690263159
    total_usage = get_hourly_usage(table, run_ts)

