#date: 2025-07-29T17:09:36Z
#url: https://api.github.com/gists/d57731f4cb90f86028f5971b6acc5de1
#owner: https://api.github.com/users/yahwang

from pyspark.conf import SparkConf
from pyspark.sql import SparkSession

NESSIE_URI='http://dev-nessie.cluster-config.svc:19120/iceberg'

conf = (
    SparkConf()
    .setAppName("Nessie")
    .set(
        "spark.jars.packages",
        "org.apache.iceberg:iceberg-aws-bundle:1.9.2,"
        "org.apache.iceberg:iceberg-spark-runtime-3.5_2.12:1.9.2," # 마지막 공백 제거
        "org.projectnessie.nessie-integrations:nessie-spark-extensions-3.5_2.12:0.104.3"
    )
    .set(
        "spark.sql.extensions",
        "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions,"
        "org.projectnessie.spark.extensions.NessieSparkSessionExtensions",
    )
    .set("spark.sql.catalog.nessie", "org.apache.iceberg.spark.SparkCatalog")
    .set("spark.sql.catalog.nessie.uri", NESSIE_URI)
    .set("spark.sql.catalog.nessie.ref", "main")
    .set("spark.sql.catalog.nessie.authentication.type", "NONE")
    .set("spark.sql.catalog.nessie.type", "rest")
    .set("spark.sql.catalog.nessie.io-impl", "org.apache.iceberg.aws.s3.S3FileIO")
    .set("spark.sql.catalog.nessie.s3.access-key-id", "...")
    .set("spark.sql.catalog.nessie.s3.secret-access-key", "...")
    .set("spark.sql.catalog.nessie.s3.endpoint", "http://minio.common.svc:80")
    .set("spark.sql.catalog.nessie.s3.region", "ap-northeast-2")
    .set("spark.sql.catalog.nessie.warehouse", "s3://nessie-warehouse/")
)

## Start Spark Session
spark = SparkSession.builder.config(conf=conf).getOrCreate()