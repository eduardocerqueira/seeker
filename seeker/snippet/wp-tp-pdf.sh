#date: 2023-11-15T17:10:23Z
#url: https://api.github.com/gists/dfff8440bb484a9ac38eb06d0afec773
#owner: https://api.github.com/users/thgie

curl -s 'https://{$DOMAIN}/wp-json/wp/v2/posts/{$ID}' | python3 -c "import sys, json; loaded = json.load(sys.stdin); output = '<h1>' + loaded['title']['rendered'] + '</h1>' + '<p><i>Published via <a href="' + loaded['link'] + '">' + loaded['link'] + '</a> on ' + loaded['date'] + '</i></p>' + loaded['content']['rendered']; print(output) ;" > content.html ; pandoc --extract-media=media content.html -t gfm-raw_html -o content.md ; sed 's/↩/back/' content.md > content_cleaned.md ; pandoc -V geometry:margin=1in content_cleaned.md -o output.pdf