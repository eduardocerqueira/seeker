#date: 2024-09-03T16:56:27Z
#url: https://api.github.com/gists/4e27bb8bae3bbff729866e1bc5866f8c
#owner: https://api.github.com/users/SeanLikesData

# Convert to geopandas
df = sedona_df.toPandas()
gdf = gpd.GeoDataFrame(df, geometry="geom")