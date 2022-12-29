#date: 2022-12-29T16:30:55Z
#url: https://api.github.com/gists/bc0b0af1f5ab48b91afae25ccd8e66bf
#owner: https://api.github.com/users/solomonvimal

#!/bin/bash
# see https://github.com/microsoft/RoadDetections

# 1) get the data
wget https://usaminedroads.blob.core.windows.net/road-detections/Oceania-Full.zip

# 2) extract the records for Australia, second column of tsv only 
zgrep AUS Oceania-Full.zip | cut -f2 > AUS.geojson

# 3) convert to GPKG, works better in QGIS
ogr2ogr -f GPKG Oceania_AUS.gpkg AUS.geojson