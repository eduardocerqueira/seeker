#date: 2022-12-28T16:45:26Z
#url: https://api.github.com/gists/d7ce9647b2ca991b2b2226cae9b96218
#owner: https://api.github.com/users/Anneysha7

affluent_neighborhoods = pd.merge(left = census_new, right = neighborhoods, on = ["NBHD_ID", "NBHD_NAME"])
affluent_neighborhoods = affluent_neighborhoods[["NBHD_ID", "NBHD_NAME", "AFFLUENT_PROP", "geometry"]]

affluent_neighborhoods["geometry"] = geopandas.GeoSeries.to_wkt(affluent_neighborhoods["geometry"])

# type(affluent_neighborhoods_gs)
affluent_neighborhoods['geometry'] = affluent_neighborhoods['geometry'].apply(wkt.loads)
affluent_neighborhoods_gdf = geopandas.GeoDataFrame(affluent_neighborhoods, geometry = affluent_neighborhoods_gs)