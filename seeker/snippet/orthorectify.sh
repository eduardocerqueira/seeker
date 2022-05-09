#date: 2022-05-09T17:01:03Z
#url: https://api.github.com/gists/fbce940599b920f8903b04e7eaf207e3
#owner: https://api.github.com/users/banesullivan

gdalwarp \
  --debug on \
  -of COG -co BLOCKSIZE=64 -co COMPRESS=DEFLATE \
  -multi \
  --config GDAL_CACHEMAX 15% -wm 15% -co NUM_THREADS=ALL_CPUS -wo NUM_THREADS=1 \
  -t_srs EPSG:4326 -et 0 -rpc -to RPC_DEM=dem.xml \
  -overwrite -srcnodata 0 -dstnodata 0 \
  source.tiff output.tiff
