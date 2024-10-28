#date: 2024-10-28T16:55:38Z
#url: https://api.github.com/gists/45e78208b3ce07d1d8490117ac9f2931
#owner: https://api.github.com/users/fehlfarbe

#!/bin/bash

find . -name "*.jpg" -type f -print0 | while read -d $'\0' file
do
		#echo "$file"
    # get resolution via jhead
		res=$(jhead "$file" 2>/dev/null | grep Resolution)
    # extract x and y resolution
	  x=$(echo $res |  awk '{print $3}')
	  y=$(echo $res |  awk '{print $5}')
		# 
		if (( $x < 1000 && $y < 1000 ))
		then
			echo "delete ${file} (${x}x${y})"
      # rm "${file}" 
		fi
done
