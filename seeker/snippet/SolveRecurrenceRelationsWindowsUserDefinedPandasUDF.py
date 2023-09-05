#date: 2023-09-05T17:00:20Z
#url: https://api.github.com/gists/3a925e1d9d1ef8fb3415ddc263c94aa0
#owner: https://api.github.com/users/dineshdharme

Not sure if this answer is helpful or not since I couldn't cast your iterative equation into a normal format or find an iterative equation solver. But you can definitely use scipy's fsolve to solve non-linear equations.

EDIT : We can used a specialized Pandas UDF to do aggregation over appropriate Window definition.

Here's an example below :

    import sys
    from pyspark import SQLContext
    from pyspark.sql.functions import *
    import pyspark.sql.functions as F
    from functools import reduce
    import pyspark
    from pyspark.sql.types import *
    import numpy as np
    from pyspark.sql.window import Window
    from scipy.optimize import fsolve
    from math import exp
    from typing import Iterator, Tuple
    import pandas as pd
    
    sc = SparkContext('local')
    sqlContext = SQLContext(sc)
    sqlContext.setConf("spark.streaming.backpressure.enabled", True)
    
    import pyspark.pandas as ps
    
    # Sample input data
    data = [(1, 1, 0.0, 1, 10),
            (1, 2, 0.5, 1, 20),
            (2, 3, 1.0, 2, 10),
            (2, 4, 1.3, 2, 20),
            (2, 5, 1.6, 2, 30),
            (3, 6, 2.0, 1, 10),
            (3, 7, 2.5, 1, 20),
            (4, 8, 3.0, 2, 10),
            (4, 9, 3.3, 2, 20),
            (4, 10, 3.6, 2, 30)]
    
    
    
    # Create a pyspark.pandas DataFrame with input columns t, v, and z
    column_list = ["group_id", "row_num", "t", "v", "z"]
    
    df_spark = sqlContext.createDataFrame(data=data, schema =column_list)
    print("Printing out df_spark")
    df_spark.show(20, truncate=False)
    
    
    
    def equations_provided(args_tuple):
        x, y = args_tuple[0], args_tuple[1]
        eq1 = x+y**2-4
        eq2 = exp(x) + x*y - 3
        return [eq1, eq2]
    
    def fsolve_call_scipy(x, y):
        ## https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fsolve.html
        result = fsolve(equations_provided,(x, y))
        python_result_list = result.tolist()
        #print(f"Input : ({x}, {y}) -- Output : {result} datatype {type(result)} python result list {result.tolist()}")
        return python_result_list
    
    solver_udf = udf(fsolve_call_scipy)
    
    answer_df = df_spark.select(*df_spark.columns, solver_udf(df_spark["t"], df_spark["v"]).alias("soln"))
    print("answer_df dataframe")
    answer_df.show(n=100, truncate=False)
    
    
    
    @pandas_udf("string")
    def multiply_aggregation(input_series: pd.Series) -> str :
    
        ### Initialization
        x_0 = 0.0
        y_0 = 1.0
    
        ## Current will be updated as we go along. At first, they will be initial values
        x_current = x_0
        y_current = y_0
    
        list_mine = []
        for ii in input_series:
            x_new = ii[0] * y_current + x_current
            y_new = y_current + 0.01 * ii[1]
    
            x_current = x_new
            y_current = y_new
    
        print("aggreated series first value is being printed lets see")
        print("list mine", list_mine)
    
        return str(x_current)+"@@"+str(y_current)
    
    
    windowSpec = Window.partitionBy("group_id").orderBy("row_num").rowsBetween(Window.unboundedPreceding, 0)
    
    df_spark = df_spark.withColumn("concat_v_z", F.array(["v", "z"]))
    
    calculated_columns_df = df_spark.withColumn("x_y_calculated", multiply_aggregation(F.col("concat_v_z")).over(windowSpec))
    print("Calculated Dataframe : Window Aggregation Function Applied")
    calculated_columns_df.show(n=100, truncate=False)
    
    calculated_columns_df = calculated_columns_df.withColumn("x_y_values_splitted", F.split("x_y_calculated", "@@"))
    calculated_columns_df = calculated_columns_df.withColumn("x_calculated", F.col("x_y_values_splitted").getItem(0).cast("float"))
    calculated_columns_df = calculated_columns_df.withColumn("y_calculated", F.col("x_y_values_splitted").getItem(1).cast("float"))
    
    print("Calculated Dataframe: Values Splitted and  Casted")
    calculated_columns_df.show(n=100, truncate=False)



Output is as follows :

    Printing out df_spark
    +--------+-------+---+---+---+
    |group_id|row_num|t  |v  |z  |
    +--------+-------+---+---+---+
    |1       |1      |0.0|1  |10 |
    |1       |2      |0.5|1  |20 |
    |2       |3      |1.0|2  |10 |
    |2       |4      |1.3|2  |20 |
    |2       |5      |1.6|2  |30 |
    |3       |6      |2.0|1  |10 |
    |3       |7      |2.5|1  |20 |
    |4       |8      |3.0|2  |10 |
    |4       |9      |3.3|2  |20 |
    |4       |10     |3.6|2  |30 |
    +--------+-------+---+---+---+
    
    answer_df dataframe
    +--------+-------+---+---+---+----------------------------------------+
    |group_id|row_num|t  |v  |z  |soln                                    |
    +--------+-------+---+---+---+----------------------------------------+
    |1       |1      |0.0|1  |10 |[0.6203445234850499, 1.838383930661961] |
    |1       |2      |0.5|1  |20 |[0.6203445234785517, 1.8383839306684822]|
    |2       |3      |1.0|2  |10 |[0.6203445234852288, 1.8383839306615979]|
    |2       |4      |1.3|2  |20 |[0.6203445234852818, 1.8383839306616214]|
    |2       |5      |1.6|2  |30 |[0.6203445234851762, 1.838383930661591] |
    |3       |6      |2.0|1  |10 |[0.6203445234852258, 1.8383839306615946]|
    |3       |7      |2.5|1  |20 |[0.620344523485133, 1.8383839306616825] |
    |4       |8      |3.0|2  |10 |[0.6203445234858643, 1.8383839306615082]|
    |4       |9      |3.3|2  |20 |[0.6203445234852366, 1.8383839306615928]|
    |4       |10     |3.6|2  |30 |[0.6203445234863979, 1.8383839306614038]|
    +--------+-------+---+---+---+----------------------------------------+
    
    
    +--------+-------+---+---+---+----------+----------------------+
    |group_id|row_num|t  |v  |z  |concat_v_z|x_y_calculated        |
    +--------+-------+---+---+---+----------+----------------------+
    |1       |1      |0.0|1  |10 |[1, 10]   |1.0@@1.1              |
    |1       |2      |0.5|1  |20 |[1, 20]   |2.1@@1.3              |
    |2       |3      |1.0|2  |10 |[2, 10]   |2.0@@1.1              |
    |2       |4      |1.3|2  |20 |[2, 20]   |4.2@@1.3              |
    |2       |5      |1.6|2  |30 |[2, 30]   |6.800000000000001@@1.6|
    |3       |6      |2.0|1  |10 |[1, 10]   |1.0@@1.1              |
    |3       |7      |2.5|1  |20 |[1, 20]   |2.1@@1.3              |
    |4       |8      |3.0|2  |10 |[2, 10]   |2.0@@1.1              |
    |4       |9      |3.3|2  |20 |[2, 20]   |4.2@@1.3              |
    |4       |10     |3.6|2  |30 |[2, 30]   |6.800000000000001@@1.6|
    +--------+-------+---+---+---+----------+----------------------+
    
    Calculated Dataframe: Values Splitted and  Casted
    +--------+-------+---+---+---+----------+----------------------+------------------------+------------+------------+
    |group_id|row_num|t  |v  |z  |concat_v_z|x_y_calculated        |x_y_values_splitted     |x_calculated|y_calculated|
    +--------+-------+---+---+---+----------+----------------------+------------------------+------------+------------+
    |1       |1      |0.0|1  |10 |[1, 10]   |1.0@@1.1              |[1.0, 1.1]              |1.0         |1.1         |
    |1       |2      |0.5|1  |20 |[1, 20]   |2.1@@1.3              |[2.1, 1.3]              |2.1         |1.3         |
    |2       |3      |1.0|2  |10 |[2, 10]   |2.0@@1.1              |[2.0, 1.1]              |2.0         |1.1         |
    |2       |4      |1.3|2  |20 |[2, 20]   |4.2@@1.3              |[4.2, 1.3]              |4.2         |1.3         |
    |2       |5      |1.6|2  |30 |[2, 30]   |6.800000000000001@@1.6|[6.800000000000001, 1.6]|6.8         |1.6         |
    |3       |6      |2.0|1  |10 |[1, 10]   |1.0@@1.1              |[1.0, 1.1]              |1.0         |1.1         |
    |3       |7      |2.5|1  |20 |[1, 20]   |2.1@@1.3              |[2.1, 1.3]              |2.1         |1.3         |
    |4       |8      |3.0|2  |10 |[2, 10]   |2.0@@1.1              |[2.0, 1.1]              |2.0         |1.1         |
    |4       |9      |3.3|2  |20 |[2, 20]   |4.2@@1.3              |[4.2, 1.3]              |4.2         |1.3         |
    |4       |10     |3.6|2  |30 |[2, 30]   |6.800000000000001@@1.6|[6.800000000000001, 1.6]|6.8         |1.6         |
    +--------+-------+---+---+---+----------+----------------------+------------------------+------------+------------+

