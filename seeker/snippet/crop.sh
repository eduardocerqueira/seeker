#date: 2024-05-02T17:03:39Z
#url: https://api.github.com/gists/382c189a8306936b94af72554e2bdd3c
#owner: https://api.github.com/users/silexcorp

#!/bin/bash

# Input image
input_image="input.png"

# Step 1: Identify Regions of Interest
# Example: Using edge detection with Canny
convert "$input_image" -canny 0x1+10%+30% "$input_image"_edges.png

# Step 2: Crop Identified Regions
# Example: Cropping based on bounding boxes
bboxArr=($(convert "$input_image"_edges.png \
-type bilevel \
-define connected-components:exclude-header=true \
-define connected-components:area-threshold=100 \
-define connected-components:mean-color=true \
-define connected-components:verbose=true \
-connected-components 8 \
null: | grep "gray(255)" | awk '{print $2}'))

num=${#bboxArr[*]}
echo "Number of items found: $num"

for ((i=0; i<num; i++)); do
    bbox="${bboxArr[$i]}"
    convert "$input_image" -crop $bbox +repage "item_$i.png"
done

echo "Cropping complete."
