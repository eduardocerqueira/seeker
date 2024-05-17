#date: 2024-05-17T17:01:09Z
#url: https://api.github.com/gists/ae99702ab98c8a629a4ca870b2873831
#owner: https://api.github.com/users/jcardus

sudo docker run -it -v nominatim \
  -flatnode:/nominatim/flatnode \
  -e PBF_URL=https://download.geofabrik.de/europe-latest.osm.pbf \
  -e REPLICATION_URL=https://download.geofabrik.de/europe-updates/ \
  -e UPDATE_MODE=catch-up
  -e REVERSE_ONLY=true \
  --shm-size=16g \
  -p 8080:8080 \
  --name nominatim \
  mediagis/nominatim:4.4 