#date: 2022-08-26T16:43:43Z
#url: https://api.github.com/gists/f7b232094265b8205f29c2865496b994
#owner: https://api.github.com/users/BlasBenito

#!/bin/bash	
#reprojects a geotif from MODIS sinusoidal to EPSG 4326
#check https://gdal.org/programs/gdalwarp.html for further details
gdalwarp \
  -of GTIFF \
  -multi \
  --config GDAL_CACHEMAX 1024 \
  -s_srs '+proj=sinu +R=6371007.181 +nadgrids=@null +wktext' \
  -t_srs '+proj=longlat +datum=WGS84 +no_defs' \
  -r near \
  -co "COMPRESS=DEFLATE" \
  input.tif \
  output.tif