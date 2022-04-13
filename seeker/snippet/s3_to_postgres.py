#date: 2022-04-13T16:58:57Z
#url: https://api.github.com/gists/734533c9fde7a93d328fa9ab2c2601df
#owner: https://api.github.com/users/PandaWhoCodes

def upload_gziped_df(bucketName, filename, df):
    csv_buffer = io.BytesIO()
    df.to_csv(csv_buffer, index=False, compression="gzip", encoding="utf-8")
    # write stream to S3
    s3 = boto3.client("s3")
    metadata = {
        "Content-Encoding": "gzip",
    }

    s3.put_object(
        Bucket=bucketName,
        Key=filename,
        Body=csv_buffer.getvalue(),
        Metadata=metadata,
        ContentEncoding="gzip",
    )


def get_lengths_for_sql(df):
    # calculate the lengths for creating tables
    lengths = df.astype("str").applymap(lambda x: len(x)).max()
    meta_dict = pd.Series(lengths.values, index=df.columns).to_dict()
    for name in meta_dict:
        meta_dict[name] = String(meta_dict[name])
    return meta_dict


def gen_table_from_df(df):
    df.head(0).to_sql(
        name=table_name,
        con=connection_detail,
        schema=schema,
        if_exists="replace",
        dtype=meta_dict,
        index=False,
    )

def s3_to_postgres(schema,table_name,bucket_name,s3_key,region):
    engine = sqlalchemy.create_engine()
    connection = engine.connect()
    sql = f"""SELECT aws_s3.table_import_from_s3 (
    '{schema}.{table_name}',  -- the table where you want the data to be imported
    '', -- column list. Empty means to import everything
    '(FORMAT csv, HEADER true)', -- this is what I use to import standard CSV
    '{bucket_name}', -- the bucket name and ONLY the bucket name, without anything else
    '{s3_key}', -- the path from the bucket to the file. Does not have to be gz
    '{region}' -- the region where the bucket is
    );"""
    cursor.execute(sql)
    connection.commit()
    cursor.close()


df = pd.read_csv("records.csv")
table_name = "test_table"
schema = "mediflex_raw"
# convert the lengths into values that can be used by sqlalchemy
meta_dict = get_lengths_for_sql(df)
# generate the table
gen_table_from_df(df)
# filename
s3_key = "test/test_file.csv.gz"
# upload the file gzipped
upload_gziped_df(bucket_name, s3_key, df)
# s3_to_postgres
s3_to_postgres(schema,table_name,bucket_name,s3_key,region)
