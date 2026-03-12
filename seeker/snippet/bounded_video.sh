#date: 2026-03-12T17:44:57Z
#url: https://api.github.com/gists/1f6e4f519134c9ed97c6ae6c6c3842fa
#owner: https://api.github.com/users/feddynventor

#!/usr/bin/env bash
set -euo pipefail

in="${1:?input file required}"
out="${2:-output.mp4}"

filter=$(
  ffprobe -v error \
    -show_entries stream=codec_type,start_time,duration \
    -of csv=p=0 "$in" |
    awk -F, '
      function num(x) {
        return (x == "" || x == "N/A") ? -1 : x + 0
      }

      $1 == "video" && !got_v {
        v_start = num($2)
        v_dur = num($3)
        got_v = 1
      }

      $1 == "audio" && !got_a {
        a_start = num($2)
        a_dur = num($3)
        got_a = 1
      }

      END {
        if (!got_v || !got_a) {
          print "missing video or audio stream" > "/dev/stderr"
          exit 1
        }

        if (v_dur < 0 || a_dur < 0) {
          print "stream duration unavailable" > "/dev/stderr"
          exit 2
        }

        v_end = v_start + v_dur
        a_end = a_start + a_dur

        overlap_start = (v_start > a_start ? v_start : a_start)
        overlap_end = (v_end < a_end ? v_end : a_end)
        overlap_dur = overlap_end - overlap_start

        if (overlap_dur <= 0) {
          print "video and audio do not overlap" > "/dev/stderr"
          exit 3
        }

        v_trim_start = overlap_start - v_start
        a_trim_start = overlap_start - a_start

        printf \
          "[0:v:0]trim=start=%.6f:duration=%.6f," \
          "setpts=PTS-STARTPTS[v];" \
          "[0:a:0]atrim=start=%.6f:duration=%.6f," \
          "asetpts=PTS-STARTPTS[a]",
          v_trim_start,
          overlap_dur,
          a_trim_start,
          overlap_dur
      }
    '
)

ffmpeg -fflags +genpts -i "$in" \
  -filter_complex "$filter" \
  -map "[v]" -map "[a]" \
  -c:v libx264 -preset slow -crf 18 -pix_fmt yuv420p \
  -c:a aac -b:a 192k \
  -map_metadata -1 -map_chapters -1 \
  -movflags +faststart \
  "$out"
