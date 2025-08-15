#date: 2025-08-15T16:56:35Z
#url: https://api.github.com/gists/a69dfcb7bdbefe7e384fbbe2668ce4d1
#owner: https://api.github.com/users/farseerfc

#!/bin/bash

for year in 2022 2023 2024 2025; do
    curl -s https://arch-archive.tuna.tsinghua.edu.cn/$year/ | grep -oE "href=\"[-0-9]+/\"" | grep -oEi "[-0-9]+" | \
        xargs -I@ bash -c "curl -s https://arch-archive.tuna.tsinghua.edu.cn/$year/@/core/os/x86_64/ | grep linux-firmware >$year-@.html"
done

for date_html in $(ls 2*.html); do
    date_name=$(basename $date_html .html)
    grep "table" $date_html >/dev/null || (echo '<table>';cat $date_html; echo '</table>') | sponge $date_html
    cha -d -c 'table {width: 100em}' $date_html | \
        grep -v '.sig ' | \
        awk -F' [ ]+' '{print "'$date_name'"","$1","$2}' > $date_name.csv
done

cat 2*.csv >linux-firmware.csv