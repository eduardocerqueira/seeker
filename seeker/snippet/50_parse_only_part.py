#date: 2022-02-07T16:59:18Z
#url: https://api.github.com/gists/4456077bbe29e3e215bde0dde9e0e654
#owner: https://api.github.com/users/kolinkorr839

# The SoupStrainer class allows you to choose which parts of an 
# incoming document are parsed
from bs4 import SoupStrainer

# conditions
only_a_tags = SoupStrainer("a")
only_tags_with_id_link2 = SoupStrainer(id="link2")

def is_short_string(string):
  return len(string) < 10
only_short_strings = SoupStrainer(string=is_short_string)

# execute parse
BeautifulSoup(html_doc, "html.parser", parse_only=only_a_tags)
BeautifulSoup(html_doc, "html.parser", parse_only=only_tags_with_id_link2)
BeautifulSoup(html_doc, "html.parser", parse_only=only_short_strings)