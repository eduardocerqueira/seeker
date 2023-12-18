#date: 2023-12-18T17:07:02Z
#url: https://api.github.com/gists/c13256200f5d582131720e2548f30e88
#owner: https://api.github.com/users/bitmvr

#!/usr/bin/env bash

# The following script assumes you haveve exported SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET
# as environment variables. To obtain these secrets, you must create a Spotify Developer
# Account. Learn more at https://developer.spotify.com/

SPOTIFY_ACCESS_TOKEN_FILE= "**********"

__request_spotify_access_token(){
  token_url='https: "**********"
  data= "**********"
       --silent \
       --header "Content-Type: application/x-www-form-urlencoded" \
       --data "grant_type=client_credentials" \
       --data "client_id=${SPOTIFY_CLIENT_ID}" \
       --data "client_secret= "**********"
  )"
  echo "$data" > "$SPOTIFY_ACCESS_TOKEN_FILE"
}

__read_spotify_access_token(){
  jq --raw-output .access_token "$SPOTIFY_ACCESS_TOKEN_FILE"
}

__has_error(){
  echo "$1" | jq --exit-status '.error' > /dev/null 2>&1
}

__print_error(){
  echo "$1" | jq --raw-output '.error.message'
}

__confirm_response(){
  response="$1"
  if __has_error "$response"; then
    __print_error "$response"
    exit 1
  fi
}

__request_artist_data(){
  access_token= "**********"
  artist_id="$2"
  response="$(curl "https://api.spotify.com/v1/artists/${artist_id}" \
                   --silent \
                   --header "Authorization: "**********"
  )"
  __confirm_response "$response" && echo "$response"
}

__request_spotify_access_token

access_token= "**********"
artist_id='4Z8W4fKeB5YxbusRsdQVPb'

__request_artist_data "$access_token" "$artist_id"
='4Z8W4fKeB5YxbusRsdQVPb'

__request_artist_data "$access_token" "$artist_id"
