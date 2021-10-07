#date: 2021-10-07T16:54:39Z
#url: https://api.github.com/gists/b633d15720cf3c94b20503f5b1129ef3
#owner: https://api.github.com/users/abxda

BD_MANZANAS = spark.read.parquet(f"../SCINCE_Parquets/*.parquet")
BD_MANZANAS_EEVVV = BD_MANZANAS.select('CVEGEO', 'ECO1_R', 'EDU46_R', 'VIV82_R', 'VIV83_R', 'VIV84_R', 'geometry')
BD_MANZANAS_EEVVV.cache()
BD_MANZANAS_EEVVV.printSchema()
BD_MANZANAS_EEVVV.show()