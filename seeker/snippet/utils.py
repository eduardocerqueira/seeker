#date: 2023-01-10T17:08:09Z
#url: https://api.github.com/gists/f09e6f3fd7acd61b929ef9233a706900
#owner: https://api.github.com/users/Shlomigreen

from simplified_scrapy.simplified_doc import SimplifiedDoc
import json
import html

with open('special_characters.json', 'r') as f:
    _special_chars =  json.load(f)
    
def convert_special_char_to_normal_char(text:str) -> str:
    """
    Convert special characters to normal characters.
    """
    target, special_list = _special_chars['base'], _special_chars['special']

    for l in special_list.values():
        for index, character in enumerate(l):
            text = text.replace(character, target[index])

    return text

def remove_html_tags(text:str) -> str:
    """
    Remove HTML tags
    """
    return SimplifiedDoc(text).text

def unescape_html(text:str) -> str:
    """
    Unescape HTML entities in the given text.
    """
    return html.unescape(text)