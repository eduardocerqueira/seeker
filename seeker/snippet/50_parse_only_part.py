#date: 2022-12-29T16:28:28Z
#url: https://api.github.com/gists/4d7e8fc8dcfaad1dee3ea671ef8c680a
#owner: https://api.github.com/users/matrixise

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