#date: 2024-04-15T16:44:45Z
#url: https://api.github.com/gists/ef8580d54820c9415c95112d1e8ca19e
#owner: https://api.github.com/users/Amb0s

#!/usr/bin/env bash

set -Eeuo pipefail
trap cleanup SIGINT SIGTERM ERR EXIT

usage() {
  cat <<EOF
Usage: $(basename "${BASH_SOURCE[0]}") [-h] [-v] [-u]

Configure PulseAudio to fix audio when screen sharing on Discord.

Available options:

-h, --help             Print this help and exit
-v, --verbose          Print script debug information
-u, --unload           Unload modules
EOF
  exit
}

cleanup() {
  trap - SIGINT SIGTERM ERR EXIT
}

msg() {
  echo >&2 -e "${1-}"
}

die() {
  local msg=$1
  local code=${2-1} # Default exit status 1.
  msg "$msg"
  exit "$code"
}

parse_params() {
  unload=false

  while :; do
    case "${1-}" in
    -h | --help) usage ;;
    -v | --verbose) set -x ;;
    -u | --unload)
      unload=true
      shift
      ;;
    -?*) die "Unknown option: $1" ;;
    *) break ;;
    esac
    shift
  done

  args=("$@")

  return 0
}

parse_params "$@"

if [ $unload && pactl list modules | grep -qF "module-loopback" && pactl list modules | grep -qF "module-null-sink" ]
then 
  pactl unload-module module-loopback
  pactl unload-module module-null-sink
else
  # Source: https://raw.githubusercontent.com/Neko-Life/Discord-Screen-Share-Fix-Audio-on-Linux/main/discord-scrshrwa.sh

  # Open pavucontrol and find your application in the "Playback" tab and set its playback on "app".
  # Then, go to the "Recording" tab and find something like WEBRTC VoiceEngine :recStream 
  # and set its source to "Monitor of mic+app".

  pactl load-module module-loopback source=@DEFAULT_SOURCE@ sink=V1
  pactl load-module module-loopback source=V2.monitor sink=V1
  pactl load-module module-loopback source=V2.monitor sink=@DEFAULT_SINK@
  pactl load-module module-null-sink sink_name=V1 sink_properties=device.description=mic+app
  pactl load-module module-null-sink sink_name=V2 sink_properties=device.description=app
fi
