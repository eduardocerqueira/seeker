#date: 2022-03-23T17:09:09Z
#url: https://api.github.com/gists/6f1e30fc4ebe23e84f1973e834b520f3
#owner: https://api.github.com/users/ninp0

#!/bin/bash
main_feature=`lsdvd | grep "Longest track" | awk '{print $3}'` # Choose the largest track for the main feature
out_file="/home/cerberus/Videos/$1.avi"
if [[ $1 != "" ]]; then
  mencoder dvd://$main_feature \
           -nosub \
           -noautosub \
           -ovc lavc \
           -lavcopts vcodec=mpeg4:vhq:vbitrate="1200" \
           -vf scale \
           -zoom \
           -xy 720 \
           -oac mp3lame \
           -lameopts br=128 \
           -alang en \
           -o $out_file
else
  echo "${0} <name_of_dvd>"
  #echo "Available Tracks are: "
  #lsdvd
fi