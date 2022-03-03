#date: 2022-03-03T16:54:07Z
#url: https://api.github.com/gists/b86fcf639b2fc469138a063a9667aa69
#owner: https://api.github.com/users/d0choa

import pyspark.sql.functions as F
from pyspark import SparkConf
from pyspark.sql import SparkSession

sparkConf = SparkConf()
sparkConf = sparkConf.set('spark.hadoop.fs.gs.requester.pays.mode', 'AUTO')
sparkConf = sparkConf.set('spark.hadoop.fs.gs.requester.pays.project.id',
                          'open-targets-eu-dev')

# establish spark connection
spark = (
    SparkSession.builder
    .config(conf=sparkConf)
    .master('local[*]')
    .getOrCreate()
)

toplociPath = "gs://genetics-portal-dev-staging/v2d/220210/toploci_betas_fixed.parquet"
ldPath = "gs://genetics-portal-dev-staging/v2d/220210/ld.parquet"
fmPath = "gs://genetics-portal-dev-staging/v2d/220210/finemapping.parquet"

toploci = spark.read.parquet(toplociPath)
fmLoci = spark.read.parquet(fmPath)
ldLoci = spark.read.parquet(ldPath)

out = (
    toploci
    .select("chrom", "pos", "ref", "alt")
    .union(
        ldLoci
        .select(
            F.col("lead_chrom").alias("chrom"),
            F.col("lead_pos").alias("pos"),
            F.col("lead_ref").alias("ref"),
            F.col("lead_alt").alias("alt")
        )
    )
    .union(
        ldLoci
        .select(
            F.col("tag_chrom").alias("chrom"),
            F.col("tag_pos").alias("pos"),
            F.col("tag_ref").alias("ref"),
            F.col("tag_alt").alias("alt")
        )
    )
    .union(
        fmLoci
        .select(
            F.col("lead_chrom").alias("chrom"),
            F.col("lead_pos").alias("pos"),
            F.col("lead_ref").alias("ref"),
            F.col("lead_alt").alias("alt")
        )
    )
    .union(
        fmLoci
        .select(
            F.col("tag_chrom").alias("chrom"),
            F.col("tag_pos").alias("pos"),
            F.col("tag_ref").alias("ref"),
            F.col("tag_alt").alias("alt")
        )
    )
    .distinct()
)