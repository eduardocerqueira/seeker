#date: 2023-07-06T16:59:32Z
#url: https://api.github.com/gists/5e6b228191fb152663e58d760d499dc9
#owner: https://api.github.com/users/johnghill

#!/bin/bash

sum_pages=0

while IFS= read -r file; do
  cur_pages=$(pdfinfo "$file" | awk '/Pages/{print $2}')
  echo "pdf: $file has $cur_pages pages."
  sum_pages=$(( sum_pages + cur_pages))
done < <(find . -name "*.pdf" -type f)

echo "All pdf files in this directory and its subdirectories have $sum_pages pages."
