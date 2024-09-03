#date: 2024-09-03T16:56:27Z
#url: https://api.github.com/gists/4e27bb8bae3bbff729866e1bc5866f8c
#owner: https://api.github.com/users/SeanLikesData

point_csv_df = sedona.read.format("csv").\
    option("delimiter", ",").\
    option("header", "false").\
    load("data/testpoint.csv")

point_csv_df.createOrReplaceTempView("pointtable")

point_df = sedona.sql("
                        select ST_Point(
                            cast(pointtable._c0 as Decimal(24,20)), 
                            cast(pointtable._c1 as Decimal(24,20))
                            ) 
                        as arealandmark from pointtable
                    ")
point_df.show(5)