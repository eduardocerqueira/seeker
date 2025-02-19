#date: 2025-02-19T16:59:03Z
#url: https://api.github.com/gists/52694e276dec62ebae24440aa25eb2f2
#owner: https://api.github.com/users/fox-io

#!/bin/bash
#
# Run this script in a folder full of ".url" files, and pipe output to an HTML file.
# Example: ./convert_url_files_to_bookmarks.sh > bookmarks.html

echo "<!DOCTYPE NETSCAPE-Bookmark-file-1>"
echo '<META HTTP-EQUIV="Content-Type" CONTENT="text/html; charset=UTF-8">'
echo '<TITLE>Bookmarks</TITLE>'
echo '<H1>Bookmarks</H1>'
echo '<DL><p>'
ls -1 *.url |
  sed 's/.url//' |
  while read L; do
    echo -n '    <DT><A HREF="';
    cat "$L.url" | grep URL | grep -v BASEURL | sed 's/URL=//' | tr -d '\r'| tr -d '\n'; echo '">'"$L"'</A>';
  done
echo "</DL><p>"
