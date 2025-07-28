#date: 2025-07-28T16:37:32Z
#url: https://api.github.com/gists/61033475c3409f8b2211c0d23daf8d90
#owner: https://api.github.com/users/FransUrbo

#!/bin/env bash

function DownloadVideo(){
    video="${1}"
    out="${2}"

    [[ -e "videos/${out}" ]] && return
    rm -f ./.tmp2

    # Get the part list.
    curl -s "${video}" -o ./.tmp1

    # Download each individual piece
    cat ./.tmp1 | grep ^http | \
        while read -r url; do
            name="$(echo "${url}" | sed -e 's@?.*@@' -e 's@.*/@@')"
            yt-dlp "$url" -o ".tmp-${name}" > /dev/null
            echo "file '.tmp-${name}'" >> ./.tmp2
        done

    # Merge the individual pieces downloaded into one video.
    mkdir -p videos
    ffmpeg -f concat -safe 0 -i ./.tmp2 -c copy "videos/${out}" > /dev/null 2>&1

    rm -f ./.tmp*
}

function GetVideos(){
  username="$1"
  page=0
  while true; do
    a=($(curl -s "https://leakedzone.com/$username?page=$page&type=videos&order=0" \
      -H 'X-Requested-With: XMLHttpRequest' | \
      jq -r '.[]|(.slug + "/" + .stream_url_play)'))
    n=0
    [[ "${#a[@]}" -eq 0 ]] && return
    for i in "${a[@]}"; do
      slug="$(cut -d "/" -f1 <<< "$i")"
      path="$(cut -d "/" -f2 <<< "$i")"
      url="$(echo -n "$path" | cut -c17- | rev | cut -c17- | base64 -d)"
      out="$(echo $url | sed -e 's@?.*@@' -e 's@.*/@@' -e 's@\..*@@')"
      DownloadVideo "${url}" "${out}.mp4"
      echo -ne "\rpage: $page, video: $n/${#a[@]}  "
      ((n++))
    done
    ((page++))
  done
}

function GetPhotos(){
  username="$1"
  page=0
  while true; do
    a=($(curl -s "https://leakedzone.com/$username?page=$page&type=photos&order=0" \
      -H 'X-Requested-With: XMLHttpRequest' | jq -r '.[].image'))
    [[ "${#a[@]}" -eq 0 ]] && return
    n=0
    for i in "${a[@]}"; do
      wget -qP photos "https://leakedzone.com/storage/$i"
      echo -ne "\rpage: $page, image: $n/${#a[@]}  "
      ((n++))
    done
    ((page++))
  done
}

GetPhotos "$1"
GetVideos "$1"