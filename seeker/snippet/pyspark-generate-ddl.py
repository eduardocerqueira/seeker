#date: 2022-03-30T16:58:32Z
#url: https://api.github.com/gists/9635918ef68f2ce0835e135e823b2d3f
#owner: https://api.github.com/users/leehuwuj

###
# HOW TO RUN
# install packages: pyspark, click 
# Submit the spark job:
# ex:
# spark-submit pyspark-generate-ddl.py --file_path sample_data.parquet --format parquet --table_name sample_table --table_loc s3://it_works/thanks_god/sample_table
##

import click
from pyspark.sql import SparkSession

DDL_CREATE_TABLE_TEMPLATE = '''
CREATE TABLE IF NOT EXISTS {table_name} (
\t{table_schema}
)
STORED AS {table_format} 
LOCATION "{table_loc}";
'''

def init_spark():
    def quiet_logs(sc):
        """Disable log"""
        logger = sc._jvm.org.apache.log4j
        logger.LogManager.getLogger("org"). setLevel( logger.Level.ERROR )
        logger.LogManager.getLogger("akka").setLevel( logger.Level.ERROR )
  
    spark = (SparkSession
        .builder
        .appName("Spark auto general DDL from data")
        .getOrCreate())
    quiet_logs(spark.sparkContext)
    return spark


@click.command()
@click.option('--file_path', help='The data directory')
@click.option('--format', help='The data format: parquet,json,csv')
@click.option('--table_name', help='The target table name')
@click.option('--table_loc', help='The external path of table')
def generate(file_path: str, format: str, table_name: str, table_loc: str):
    # Init new spark session
    spark = init_spark()

    # Load data
    if format == 'parquet':
        data = spark.read.parquet(file_path)
    elif format == 'json':
        data = spark.read.json(file_path)
    elif format == 'csv':
        data = spark.read.csv(file_path)
    else:
        raise Exception("Data format is not supported!")
    
    # Get table schema from spark
    table_schema = data._jdf.schema().toDDL().replace(',`', ',\n\t`')

    # Generate DDL query
    target_query = DDL_CREATE_TABLE_TEMPLATE.format(
        table_name=table_name,
        table_schema=table_schema,
        table_format=format.upper(),
        table_loc=table_loc
    )

    print(target_query)

    return target_query


if __name__ == '__main__':
    generate()
    pass