#date: 2022-08-26T16:51:56Z
#url: https://api.github.com/gists/cda94ac1970ab3503e0726e02c3326f2
#owner: https://api.github.com/users/BlasBenito

#!/bin/bash	
#reprojects in parallel geotif files in MODIS sinusoidal 
#in the "raw" folder
#to geotif files in EPSG 4326 in the current folder
#with the suffix "_wgs_84"

#listing files in raw
for FILEPATH in raw/*; do

  #generates output file name
  FILENAME=$(basename -- "$FILEPATH")
  FILEOUT="${FILENAME%.*}"_wgs_84.tif

  #applies gdal warp in parallel
  gdalwarp \
    -of GTIFF \
    --config GDAL_CACHEMAX 1024 \
    -multi -wo "NUM_THREADS=ALL_CPUS" \
    -s_srs '+proj=sinu +R=6371007.181 +nadgrids=@null +wktext' \
    -t_srs '+proj=longlat +datum=WGS84 +no_defs' \
    -tr 0.008333333300000 -0.008333333300000\
    -r near \
    -co "COMPRESS=DEFLATE" \
    ${FILEPATH} \
    ${FILEOUT} \
    &

done

wait