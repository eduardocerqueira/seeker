#date: 2024-09-03T16:56:27Z
#url: https://api.github.com/gists/4e27bb8bae3bbff729866e1bc5866f8c
#owner: https://api.github.com/users/SeanLikesData

# code snippets for generating polygon tables by reading various file types.
# Note that the only real change is the ST_GeomFrom{filetype} function in line 7.

# ST_GeomfromText
polygon_wkt_df = sedona.read.format("csv").\
    option("delimiter", "\t").\
    option("header", "false").\
    load("data/county_small.tsv")

polygon_wkt_df.createOrReplaceTempView("polygontable")
polygon_df = sedona.sql("select polygontable._c6 as name, ST_GeomFromText(polygontable._c0) as countyshape from polygontable")
polygon_df.show(5)

# ST_GeomFromWKB
polygon_wkb_df = sedona.read.format("csv").\
    option("delimiter", "\t").\
    option("header", "false").\
    load("data/county_small_wkb.tsv")

polygon_wkb_df.createOrReplaceTempView("polygontable")
polygon_df = sedona.sql("select polygontable._c6 as name, ST_GeomFromWKB(polygontable._c0) as countyshape from polygontable")
polygon_df.show(5)

# ST_GeomFromGeoJSON
polygon_json_df = sedona.read.format("csv").\
    option("delimiter", "\t").\
    option("header", "false").\
    load("data/testPolygon.json")

polygon_json_df.createOrReplaceTempView("polygontable")
polygon_df = sedona.sql("select ST_GeomFromGeoJSON(polygontable._c0) as countyshape from polygontable")
polygon_df.show(5)
