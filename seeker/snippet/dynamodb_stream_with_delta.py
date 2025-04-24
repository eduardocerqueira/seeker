#date: 2025-04-24T17:04:44Z
#url: https://api.github.com/gists/247d8bcb261768dc4fa27e1a3ac52154
#owner: https://api.github.com/users/Lucasnobrepro

import boto3
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

from pyspark.sql import SparkSession
from delta import configure_spark_with_delta_pip

# ⚙️ Spark + Delta setup
builder = (
    SparkSession.builder.appName("DynamoDBStreamToDelta")
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
)

spark = configure_spark_with_delta_pip(builder).getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

# AWS clients
dynamodb = boto3.client('dynamodb')
streams = boto3.client('dynamodbstreams')

TABLE_NAME = 'prod_usar'
MAX_WORKERS = 10
DELTA_PATH = "s3://seu-bucket/dynamodb-delta-table/"

def get_stream_arn():
    res = dynamodb.describe_table(TableName=TABLE_NAME)
    return res['Table']['LatestStreamArn']

def get_shards(stream_arn):
    response = streams.describe_stream(StreamArn=stream_arn)
    return response['StreamDescription']['Shards']

def convert_dynamodb_image(image):
    # Converte o formato DynamoDB para dicionário comum
    from boto3.dynamodb.types import TypeDeserializer
    deserializer = TypeDeserializer()
    return {k: deserializer.deserialize(v) for k, v in image.items()}

def process_shard(shard, stream_arn):
    shard_id = shard['ShardId']
    iterator_args = {
        'StreamArn': stream_arn,
        'ShardId': shard_id,
        'ShardIteratorType': 'TRIM_HORIZON'
    }

    try:
        iterator = streams.get_shard_iterator(**iterator_args)['ShardIterator']
    except Exception as e:
        print(f"❌ Iterator erro: {shard_id}: {e}")
        return []

    collected = []

    while iterator:
        try:
            response = streams.get_records(ShardIterator=iterator, Limit=1000)
        except Exception as e:
            print(f"⚠️ Leitura erro no shard {shard_id}: {e}")
            break

        records = response['Records']
        for record in records:
            if record['eventName'] in ['INSERT', 'MODIFY']:
                image = record['dynamodb'].get('NewImage', {})
                deserialized = convert_dynamodb_image(image)
                deserialized['event_type'] = record['eventName']
                deserialized['shard_id'] = shard_id
                deserialized['timestamp'] = record['dynamodb']['ApproximateCreationDateTime'].isoformat()
                collected.append(deserialized)

        iterator = response.get('NextShardIterator')
        time.sleep(0.3)

    return collected

def main():
    stream_arn = get_stream_arn()
    shards = get_shards(stream_arn)

    print(f"📦 {len(shards)} shards encontrados")

    all_records = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(process_shard, shard, stream_arn) for shard in shards]
        for future in as_completed(futures):
            all_records.extend(future.result())

    if all_records:
        df = spark.createDataFrame(all_records)
        df.write.format("delta").mode("append").save(DELTA_PATH)
        print(f"✅ {len(all_records)} registros salvos em {DELTA_PATH}")
    else:
        print("ℹ️ Nenhum dado novo para salvar.")

if __name__ == "__main__":
    main()
