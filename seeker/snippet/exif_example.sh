#date: 2022-01-20T17:00:12Z
#url: https://api.github.com/gists/dc8828c1574ddf61f286ac44462bb5c5
#owner: https://api.github.com/users/kis9a

## identify image
function identify_image() {
  if is_exists "identify"; then
    identify -verbose "$1"
  else
    echo 'install imagemagick'
  fi
}

function identify_image_google_map() {
  query="$(cat "$1" |identify -verbose -|grep -e 'GPSL.*,'|tr -d ' '|awk -F '[:/,]' '$0=$3/$4+$5/$6/60+$7/$8/3600'|xargs|tr ' ' ,)"
  if [[ -n "$query" ]]; then
    echo "https://www.google.com/maps/search/?api=1&query=$query"
  fi
}

function identify_image_google_map_distribute() {
  ls *(png|jpg|jpeg|webp) | xargs -i identify -verbose {} | grep -e 'GPSL.*,'|tr -d ' '|awk -F '[:/,]' '$0=$3/$4+$5/$6/60+$7/$8/3600'|xargs -n2|tr ' ' ,|tr \\n /|xargs -i echo https://www.google.co.jp/maps/dir/{}
}

function remove_exif() {
  exiftool -all= "$1"
  # convert -strip
  # exiftool -overwrite_original -geotag=
}