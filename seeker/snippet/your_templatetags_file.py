#date: 2021-11-26T17:13:49Z
#url: https://api.github.com/gists/70cb5539d5af597c0a46ab498885ee5c
#owner: https://api.github.com/users/hardcodd

# modulename/templatetags/your_templatetags_file.py
import math
import re

from django import template


register = template.Library()


@register.filter
def set_page(uri, number):
    """Sets `page` property to uri string"""

    if '?page=' in uri or '&page=' in uri:
        uri = re.sub(r'page=[\d]+', f'page={number}', uri)
    elif re.search(r'/?[\w]+=', uri):
        uri = f'{uri}&page={number}'
    else:
        uri = f'?page={number}'

    return uri


@register.filter
def remove_page(uri):
    """Removes `page` property from uri string"""

    if '?page=' in uri or '&page=' in uri:
        uri = re.sub(r'[?&]page=[\d]+', '', uri)
    if re.search('/&', uri):
        uri = re.sub('/&', '/?', uri)

    return uri


@register.filter(name='round')
def roundto(number, to=100):
    return int(math.ceil(number / to)) * to

