#date: 2024-04-12T16:49:45Z
#url: https://api.github.com/gists/95e545aa67ce227774bfe24a06425517
#owner: https://api.github.com/users/mluis7

#!/bin/bash
#
# Convert srt subtitle file timestamps from NTSC 23.976 fps to PAL 25 fps and viceversa
# A typical srt subtitle frame looks like
#
# 1
# 00:02:13,720 --> 00:02:16,644
# - ¿Cariño?
# - Hola.

#
# NTSC sub time to PAL sub time, multiply by: (24000 / 1001) / 25000 = 0.95904095904095904095904095904096
# PAL sub time to NTSC sub time, multiply by: 25000 / (24000 / 1001) = 1.0427083333333333333333333333333

usage(){
  echo -e "Usage: ./convert-srt-fps.sh filename.srt [ntsc2pal|pal2ntsc] > converted.srt\n" 
}

if [ "$1" = "-h" ]; then
  usage
  exit
fi 

if [ ! -f "$1" ];then
  echo "ERROR: File $1 does not exists." >&2
  exit 1
fi
if [ -n "$2" ] && [ "$2" == 'ntsc2pal' ];then
  conv_rate="0.95904095904095904095904095904096"
elif [ -n "$2" ] && [ "$2" == 'pal2ntsc' ];then
  conv_rate="1.0427083333333333333333333333333"
else
  echo "ERROR. Unknown conversion '$2'. You must specify one of ntsc2pal|pal2ntsc"
  usage >&2
  exit 2
fi

marker=0
while read line; do
  # subtitle index line preceding timestamps line
  if [ "$(tr -s '[:digit:]' 'x' <<<"$line")" == 'x' ];then
    echo "$line"
    marker=1
    continue
  fi
  if [ "$marker" == 1 ];then
    read t1 n t2 <<<"$(tr ',' '.' <<<"$line")"
    # timestamp to seconds.millis
    ts1=$(date -d "1970-01-01 $t1 -00" '+%s.%3N')
    # apply conversion factor
    ts1=$(echo "scale=3; $ts1 * $conv_rate" | bc)
    # convert back to timestamp
    ts1d=$(TZ='UTC' date -d @$ts1 '+%H:%M:%S,%3N')

    ts2=$(date -d "1970-01-01 $t2 -00" '+%s.%3N')
    ts2=$(echo "scale=3; $ts2 * $conv_rate" | bc)
    ts2d=$(TZ='UTC' date -d @$ts2 '+%H:%M:%S,%3N')
    
    # print converted timestamp
    printf "%s --> %s\n" "$ts1d" "$ts2d"
    marker=0
  else
    if [ "$marker" == 0 ] && [ "$line" == '' ];then
      next=1
    fi
    echo "$line"
  fi
done < "$1"
  
  
