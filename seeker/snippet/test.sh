#date: 2022-03-17T17:05:09Z
#url: https://api.github.com/gists/28d614470c958f8815ff3f3342e2fea9
#owner: https://api.github.com/users/julianorchard

function does_image_need_resizing() {
  # Image File
    image=$1
  # Split X:X Into Two Variables (d_1 and d_2)
    r=(${2//\:/ })
    d_1=${r[0]}
    d_2=${r[1]}
  # Get Current Aspect Ratio
    c_r=$(magick convert "${image}" -format "%[fx:(w/h)]" info:)
  # Check If Outside Bounds
    r_1=$(awk "BEGIN { print $d_1 / $d_2 }") # ratio of 1/2
    r_2=$(awk "BEGIN { print $d_2 / $d_1 }") # ratio of 2/1
    if awk "BEGIN {exit (${c_r} < ${r_1})}" || awk "BEGIN {exit (${c_r} > ${r_2})}" 
    then
    # Returns Image Rightside Up
      awk "BEGIN {exit (${c_r} < 1)}" && echo "$d_1:$d_2" || echo "$d_2:$d_1"
    else 
    # Image Does Not Need Resizing
      echo 0
    fi
}

# Example Usage: 
# [[ $ratio != 0 ]] && magick convert "$nf" -gravity center -crop "$ratio" "$nf"
