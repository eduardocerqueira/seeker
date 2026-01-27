#date: 2026-01-27T16:56:38Z
#url: https://api.github.com/gists/453a1736276b3287088955546ef720f5
#owner: https://api.github.com/users/wjlafrance

#!/usr/bin/env bash
# md2styledhtml.sh
# Convert Markdown file into HTML that *looks* like Markdown, but with interactive styling.

if [ $# -lt 1 ]; then
  echo "Usage: $0 input.md"
  exit 1
fi

INPUT="$1"
OUTPUT="${INPUT%.md}.html"

# Convert Markdown to JSON AST using pandoc
# Then run a custom filter (here with jq) to wrap symbols.
# Finally wrap with HTML header+footer and CSS.

HTML=$(pandoc "$INPUT" -t html)

cat > "$OUTPUT" <<EOF
<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <title>$INPUT</title>
  <style>
    body {
      background-color: #ededed;
      width: 100%;
      max-width: 60em;
      font-family: monospace;
    }
    .md-symbol {
      color: #555;
    }
    strong { font-weight: bold; }
    em { font-style: italic; }
    a {
      color: blue;
      text-decoration: underline;
    }
ul {
  list-style: none;        /* remove default bullets */
  padding-left: 1.2em;     /* give space for our custom "-" */
}

ul li::before {
  content: "-";            /* show a dash like in Markdown */
  color: #555;             /* faded, like other md-symbols */
  margin-right: 0.5em;     /* space between dash and text */
}

blockquote {
  position: relative;
  margin: 1em 0;
  padding-left: 2em;       /* space for the marker */
  color: #444;
}

blockquote::before {
  content: ">";
  position: absolute;
  left: 0;
  top: 0;
  font-size: 2em;          /* 2× normal size */
  line-height: 1;
  color: #888;             /* faded like md-symbols */
}

  </style>
</head>
<body>
EOF

# Now we massage the HTML to re-insert Markdown symbols
#echo "$HTML" \
#  | sed -z -E 's|<strong>(.*?)</strong|<span class="md-symbol">**</span><strong>\1</strong><span class="md-symbol">**</span|g' \
#  | sed -z -E 's|<em>(.*?)</em|<span class="md-symbol">_</span><em>\1</em><span class="md-symbol">_</span|g' \
#  | sed -z -E 's|<a\s+href=\"([^\"]+)\">([^\<]+)</a>|<span class="md-symbol">[</span><a href="\1">\2</a><span class="md-symbol">]</span><span class="md-symbol">(</span>\1<span class="md-symbol">)</span>|g' \
#  | sed -z -E 's|<h1([^\>]*)?>(.*?)</h1>|<h1\1><span class="md-symbol">#</span> \2</h1>|g' \
#  | sed -z -E 's|<h2([^\>]*)?>(.*?)</h2>|<h2\1><span class="md-symbol">##</span> \2</h2>|g' \
#  | sed -z -E 's|<h3([^\>]*)?>(.*?)</h3>|<h3\1><span class="md-symbol">###</span> \2</h3>|g' \
#  >> "$OUTPUT"

echo "$HTML"  | sed -z -E '
  s|<strong>(.*?)</strong|<span class="md-symbol">**</span><strong>\1</strong><span class="md-symbol">**</span|g;
  s|<em>(.*?)</em|<span class="md-symbol">_</span><em>\1</em><span class="md-symbol">_</span|g;
  s|<a\s+href=\"([^\"]+)\">([^\<]+)</a>|<span class="md-symbol">[</span><a href="\1">\2</a><span class="md-symbol">]</span><span class="md-symbol">(</span>\1<span class="md-symbol">)</span>|g;
  s|<h1([^\>]*)?>([^\<]*?)</h1>|<h1\1><span class="md-symbol">#</span> \2</h1>|g;
  s|<h2([^\>]*)?>([^\<]*?)</h2>|<h2\1><span class="md-symbol">##</span> \2</h2>|g;
  s|<h3([^\>]*)?>([^\<]*?)</h3>|<h3\1><span class="md-symbol">###</span> \2</h3>|g;
  '>> "$OUTPUT"

cat >> "$OUTPUT" <<EOF
</body>
</html>
EOF

echo "Wrote $OUTPUT"