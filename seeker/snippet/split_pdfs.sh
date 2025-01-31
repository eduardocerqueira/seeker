#date: 2025-01-31T17:03:47Z
#url: https://api.github.com/gists/5d0d3aae7375f164a4582da7da361c4e
#owner: https://api.github.com/users/cr3a7ure

#!/bin/bash

# Split PDF files into multiple files with a specified number of pages.
# Usage: ./split_pdf.sh [increment] [directory]
#

increment=${1:-5}
directory=${2:-"."}

for file in "$directory"/*; do

  max_page_num=$(pdftk "$file" dump_data | grep NumberOfPages | awk '{print $2}')
  echo "> Total number of pages: $max_page_num"
  echo "> Number of pages per output file(s): $increment"

  if [ "$max_page_num" -lt "$increment" ]; then
      echo "> Skip split"
      continue
  fi

  for i in $(seq 1 "$increment" "$max_page_num"); do
      num_first=$i
      num_second=$(expr "$i" + "$increment" - 1)
      echo "> Splitting pages: $num_first to $num_second"
      if [ "$num_second" -gt "$max_page_num" ]; then
          num_second=$max_page_num
      fi
      idx=$(expr "$i" / "$increment")
      pdftk "$file" cat "$num_first"-"$num_second" output "${file%.pdf}-Part-${idx}.pdf"
  done
  echo "> PDF Split: ${file} Done."

done
echo "Done."
