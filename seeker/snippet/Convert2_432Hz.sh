#date: 2021-10-25T17:16:07Z
#url: https://api.github.com/gists/c232a70e081558dfcd6c797f991009df
#owner: https://api.github.com/users/ManuelTS

  GNU nano 5.4                                                                        /home/hl/Documents/Skripte/Convert2_432Hz.sh                                                                          I L    
#!/bin/bash
# Batch convert all .mp3 files in the current directory to 432Hz with ffmpeg

oldIFS=$IFS
IFS=$'\n'
set -f # Deal with blanks and special characters in file names of the file command and in for loop

found=($(find . -name "*.mp3")) # Find the files in the current directory

IFS=$oldIFS
set +f

for file in "${found[@]}"; do # Iterate the found files 
  # Math: 
  # We want to convert from 441Hz to 432Hz, the difference is 90Hz which are 2,040816327 % of 441Hz.
  # Parameters:
  # asetrate contains the desired pitch (frequency) but changes the playback speed
  # atempo ajusts the resulting playback speed change of asetrate back to normal by making the output 2,040816327 % faster
  # aresample keeps the pitch change by correct rate of (re)sampling
  mv "$file" "$file.tmp"
  ffmpeg -loglevel 8 -i "$file.tmp" -af asetrate=43200,atempo=1.02040816327,aresample=43200 "$file"
  rm "$file.tmp"
  echo "Pitched $file"
done
