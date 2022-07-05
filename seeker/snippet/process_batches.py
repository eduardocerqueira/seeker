#date: 2022-07-05T17:09:38Z
#url: https://api.github.com/gists/060da78c5c468c37e8ec3e85200f41ad
#owner: https://api.github.com/users/kevinjnguyen

import base64
import json
import pandas

def lambda_handler(event, context):
  try:
    records = event['Records']
    record_batch = []
    for record in records:
      kinesis_record = record['kinesis']
      record_data = kinesis_record['data']
      record_data = base64.b64decode(record_data).decode("utf-8")
      record = json.loads(record_data)
      print('Fetched Kinesis Record: {}'.format(record))
      ...
      # pre-process the record
      ...
      record_batch.append(record)

    df = pandas.df(record_batch)
    df.to_parquet('s3://some_s3')
  except Exception as e:
      print('Exception occurred: {}'.format(e))
      traceback.print_exc()