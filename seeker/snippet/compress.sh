#date: 2025-02-07T16:45:19Z
#url: https://api.github.com/gists/22aae69254a0472496ade5e41175cbc1
#owner: https://api.github.com/users/joshua-koehler

#!/bin/bash
bitrate=36
if [[ -n $2 ]] 
  then
    bitrate=$2
  fi
inputfilename="$1"
outputfilename="${inputfilename%.mp3}.bitrate${bitrate}.mp3"
echo "Compressing into $outputfilename"
ffmpeg -i $inputfilename -b:a ${bitrate}k -ac 1 $outputfilename