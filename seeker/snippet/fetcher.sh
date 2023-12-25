#date: 2023-12-25T16:46:29Z
#url: https://api.github.com/gists/9f031d76115d5f280c9830aa2fafe5f9
#owner: https://api.github.com/users/miixel2

#!/bin/bash

SITE="https://ro.gnjoy.in.th"
first_year=2020
current_year=$(date +%Y)

for year in $(seq $first_year $current_year); do
  filename="ROGGT_Event_${year}.md"
  echo "Processing year $year..."
  for pageNo in $(seq 1 50); do
    pageUrl="${SITE}/${year}/page/${pageNo}"

    # check if the URL exists using wget --spider
    if ! wget --spider -q "$pageUrl"; then
      echo "Ended of ${year}, skipping to next year..."
      break  # if the URL does not exist, break the loop and skip to the next year
    fi

    echo "Processing ${pageUrl}"
    echo -e "\\n## [${year} Page $pageNo]($pageUrl)" | tee -a "$filename"

    IFS=$'\n'
    itemBlocks=$(wget -qO- "$pageUrl" | grep -oP '(?s)<h2 class="entry-title"><a.*?</h2>')

    # separate blocks into array
    itemBlocks=($itemBlocks)

    for (( idx=0 ; idx<${#itemBlocks[@]} ; idx++ )); do
      block="${itemBlocks[$idx]}"
      itemUrl=$(echo "$block" | grep -oP '(?<=href=")[^"]+')
      itemTitle=$(echo "$block" | awk -F'[<>]' '/<a href=/{print $5}')

      echo "- [$itemTitle]($itemUrl)" | tee -a "$filename"
    done
  done  
done