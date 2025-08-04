#date: 2025-08-04T17:17:22Z
#url: https://api.github.com/gists/82e8c2e4854af984d2cc6a4c04a0d04c
#owner: https://api.github.com/users/toby-bro

#!/bin/bash
TEAM=00001 #REPLACE WITH THE TEAM NUMBER THAT INTERESTS YOU
table=$(curl -s https://ctftime.org/team/${TEAM} | grep -P '<div class="tab-pane active" id="rating_202\d"' -A 999 | grep -P '<div class="tab-pane" id="rating_202\d"' -B 999 | grep -P '<tr><td class="place_ico"></td><td class="place">\d+</td><td><a href="/event/\d+">[\s\w\d]+</a></td><td>[\d\.]+</td><td>[\d\.]+</td></tr>'); for i in $(echo $table | grep -Po '(?<=<td>)[\d\.]+(?=</td></tr>)' | sort -h | tail -n 10) ; do echo -n "$i -- "; echo $table | grep $i | grep -Po '(?<=<a href="/event/\d{4}">)[\w\s\d]+(?=</a>)' ; done