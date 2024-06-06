#date: 2024-06-06T17:04:45Z
#url: https://api.github.com/gists/cb919327f2866f9bbee90e54141ce032
#owner: https://api.github.com/users/tabrezm

from pyspark.sql import Column, DataFrame, SparkSession
from pyspark.sql.column import _to_java_column

def from_xml(
    xml_str_col: Column | str, schema: T.StructType, options: dict[str, str] = None
) -> Column:
    """
    Returns a column containing struct values parsed from `xml_str_col` using `schema`.
    """
    spark = SparkSession._instantiatedSession

    if isinstance(xml_str_col, Column):
        column_name, xml_str_col = (
            xml_str_col._jc.toString(),
            xml_str_col.cast("string"),
        )
    else:
        column_name, xml_str_col = xml_str_col, F.col(xml_str_col).cast("string")

    java_column = _to_java_column(xml_str_col)
    java_schema = spark._jsparkSession.parseDataType(schema.json())
    scala_options = spark._jvm.PythonUtils.toScalaMap(options)
    jc = spark._jvm.com.databricks.spark.xml.functions.from_xml(
        java_column, java_schema, scala_options
    )
    return Column(jc).alias(column_name)