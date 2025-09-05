#date: 2025-09-05T17:06:30Z
#url: https://api.github.com/gists/9c44f5a84326694683071b3896f792cc
#owner: https://api.github.com/users/danglingptr0x0

#!/bin/sh

# sway, waybar, streamlink, and mpv are required

STREAMER="emiru"

s="$(curl -fsSL --max-time 4 --retry 1 "https://decapi.me/twitch/uptime/${STREAMER}" 2>/dev/null)" || s="__ERR__"
s="$(printf '%s' "$s" | tr -d '\r' | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//' -e 's/%$//')"

if printf '%s' "$s" | grep -Eq '^[[:space:]]*[0-9]+[[:space:]]*(day|hour|minute|second)s?(,[[:space:]]*[0-9]+[[:space:]]*(day|hour|minute|second)s?)*[[:space:]]*$'; then
  state="online"
  abbrev="$(printf '%s' "$s" | sed -E -e 's/([0-9]+)[[:space:]]*day(s)?/\1d/g' -e 's/([0-9]+)[[:space:]]*hour(s)?/\1h/g' -e 's/([0-9]+)[[:space:]]*minute(s)?/\1m/g' -e 's/([0-9]+)[[:space:]]*second(s)?/\1s/g' -e 's/[[:space:]]*,[[:space:]]*//g' -e 's/[[:space:]]+//g')"
  out="${STREAMER}: $abbrev"
elif printf '%s' "$s" | grep -qi 'offline'; then
  state="offline"
  out="${STREAMER}: off"
else
  state="unknown"
  out="${STREAMER}: N/A"
fi

prev="$(cat /dev/shm/${STREAMER}.prev 2>/dev/null || printf unknown)"

if [ "$prev" != "online" ] && [ "$state" = "online" ]; then
  notify-send -u critical "${STREAMER} is LIVE!!!" "$abbrev"
  d="$HOME/${STREAMER}/twitch/live"
  mkdir -p "$d"
  f="$d/${STREAMER}_$(date +%F_%H-%M-%S).ts"
  streamlink --retry-open 9999 --retry-streams 60 "https://www.twitch.tv/${STREAMER}" best --player mpv --player-args="--force-window=immediate --title=${STREAMER} --mute=yes --ontop --no-border --no-config --no-resume-playback --cache=no --screenshot-directory=$HOME/${STREAMER}/twitch/pics/ --geometry=637x359+2775+5" --record "$f" >/dev/null 2>&1 &

  for _ in $(seq 1 100000); do id="$(swaymsg -t get_tree | jq -r '.. | objects? | select(.app_id? == "mpv" and .name? == "'"${STREAMER}"'") | .id' | head -n1)"; [ -n "$id" ] && swaymsg "[con_id=$id] floating enable" && swaymsg "[con_id=$id] sticky enable" && swaymsg "[con_id=$id] move position 2775 px 5 px" && swaymsg "[con_id=$id] move scratchpad" && break; sleep 0.001; done
fi

printf '%s\n' "$state" > /dev/shm/${STREAMER}.prev

case "$1" in
  online) [ "$state" = "online" ] && printf '%s\n' "$out" ;;
  offline) [ "$state" != "online" ] && printf '%s\n' "$out" ;;
  *) printf '%s\n' "$out" ;;
esac
