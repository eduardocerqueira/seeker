#date: 2023-08-15T16:46:04Z
#url: https://api.github.com/gists/b8a32f005d507ae07733c9ccbcd7340f
#owner: https://api.github.com/users/mk-pmb

#!/bin/bash
# -*- coding: UTF-8, tab-width: 2 -*-
# Origin: https://gist.github.com/mk-pmb/b8a32f005d507ae07733c9ccbcd7340f
# License: CC-0
echo "Content-Type: text/html; charset=UTF-8"
echo
echo '<!DOCTYPE html>'
echo '<html><head>'
echo '  <meta charset="UTF-8">'
echo '  <title>apache-cgi-dumpenv</title>'
echo '</head><body>'
echo '<pre>'
env | sort -V | LANG=C sed -re '
  s~&~\&amp;~g
  s~<~\&lt;~g
  s~>~\&gt;~g
  s~"~\&quot;~g
  '
echo '</pre>'
echo
echo '<h4>JavaScript:</h4>'
echo '<dl>'
for V in navigator.{appName,userAgent} document.{URL,cookie}; do
  echo -n "  <dt>$V:</dt><dd><script" 'type="text/javascript">'
  echo "document.write($V || '–');</script></dd>"
done
echo '</dl>'