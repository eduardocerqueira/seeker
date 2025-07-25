#date: 2025-07-25T17:06:55Z
#url: https://api.github.com/gists/f2c6b76e4a3f00cf25abec795ce78665
#owner: https://api.github.com/users/nicktelford

#!/usr/bin/env bash
#
# A streamlink wrapper that:
# 1. Only supports MPV
# 2. Launches MPV immediately, before pre-roll ads have played.
# 3. Displays a notification when ads are rolling.

# EDIT THIS LINE
# Set this to the path to an image to display when starting/loading a Twitch stream
LOADING_SCREEN="/path/to/twitch-loading-screen.png"

set -eEuo pipefail # "strict" mode (http://redsymbol.net/articles/unofficial-bash-strict-mode/)
export SHELLOPTS  # propagate strict mode to all subshells

# absolute path of the directory the invoked script resides in
BASE_PATH="${BASE_PATH:-$(dirname -- "$(readlink -f "${BASH_SOURCE[0]}")")}"

out=$(mktemp --tmpdir=/tmp -u "streamlink-twitch-XXXX.fifo")

# kill child processes when this one exits/is interrupted
trap 'pkill -TERM -P $$ && rm -f $out' EXIT

ARGS=""
PLAYER_ARGS=""

# shortcut to print streamlink version
# needed by streamlink-twitch-gui for version checks
if [ "$1" == "--version" ]; then
  streamlink --version
  exit $?
fi

# parse arguments being passed to streamlink
# 1. Strip out --player, we're using forcing stdout streaming to MPV
# 2. Extract --player-args, to pass to MPV ourselves
# 3. Retain all other arguments to pass to streamlink
while [[ $# -gt 0 ]]; do
  case $1 in
    --player|-p|--stdout)
      # override, we're forcing using stdout
      shift
      shift
      ;;
    --player-args|-a)
      # store the player args to pass directly to MPV
      # sadly, we need the eval here to handle proper quoting
      eval PLAYER_ARGS=("$2")
      shift
      shift
      ;;
    *)
      ARGS="$ARGS $1"
      shift
      ;;
  esac
done

# named pipe to capture log output from streamlink
# streamlink-twitch-gui wants it on stdout, but streamlink is using stdout to stream to mpv
# so instead, we capture it to a named pipe, and then output _that_ from our script
mkfifo "$out"
stdbuf -o0 cat "$out" &

# we need to print this to emulate streamlink launching mpv via the "--player" option
# streamlink-twitch-gui checks for this to determine if the stream launched successfully
echo "Starting player: mpv"

streamlink --stdout --logfile "$out" $ARGS | \
  mpv "${PLAYER_ARGS[@]}" \
    --image-display-duration=1 \
    --demuxer-lavf-o=timeout=100000 \
    --loop-playlist=inf \
    "$LOADING_SCREEN" - >/dev/null 2>/dev/null
