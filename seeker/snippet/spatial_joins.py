#date: 2024-09-03T16:56:27Z
#url: https://api.github.com/gists/4e27bb8bae3bbff729866e1bc5866f8c
#owner: https://api.github.com/users/SeanLikesData

# Distance Join

point_csv_df_1 = sedona.read.format("csv").\
    option("delimiter", ",").\
    option("header", "false").load("data/testpoint.csv")

point_csv_df_1.createOrReplaceTempView("pointtable")

point_df1 = sedona.sql("SELECT ST_Point(cast(pointtable._c0 as Decimal(24,20)),cast(pointtable._c1 as Decimal(24,20))) as pointshape1, \'abc\' as name1 from pointtable")
point_df1.createOrReplaceTempView("pointdf1")

point_csv_df2 = sedona.read.format("csv").\
    option("delimiter", ",").\
    option("header", "false").load("data/testpoint.csv")

point_csv_df2.createOrReplaceTempView("pointtable")
point_df2 = sedona.sql("select ST_Point(cast(pointtable._c0 as Decimal(24,20)),cast(pointtable._c1 as Decimal(24,20))) as pointshape2, \'def\' as name2 from pointtable")
point_df2.createOrReplaceTempView("pointdf2")

distance_join_df = sedona.sql("select * from pointdf1, pointdf2 where ST_Distance(pointdf1.pointshape1,pointdf2.pointshape2) < 2")
distance_join_df.explain()
distance_join_df.show(5)

# Range Join 

import pandas as pd
gdf = gpd.read_file("data/gis_osm_pois_free_1.shp")
gdf = gdf.replace(pd.NA, '')
osm_points = sedona.createDataFrame(
    gdf
)
osm_points.createOrReplaceTempView("points")

transformed_df = sedona.sql(
    """
        SELECT osm_id,
               code,
               fclass,
               name,
               ST_Transform(geometry, 'epsg:4326', 'epsg:2180') as geom 
        FROM points
    """)

transformed_df.createOrReplaceTempView("points_2180")

neighbours_within_1000m = sedona.sql("""
        SELECT a.osm_id AS id_1,
               b.osm_id AS id_2,
               a.geom 
        FROM points_2180 AS a, points_2180 AS b 
        WHERE ST_Distance(a.geom,b.geom) < 50
    """)

# Convert to geopandas
df = neighbours_within_1000m.toPandas()
gdf = gpd.GeoDataFrame(df, geometry="geom")