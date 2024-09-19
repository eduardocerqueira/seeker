#date: 2024-09-19T17:11:43Z
#url: https://api.github.com/gists/62c70fe353e74793cdd8c7bb0bd7b480
#owner: https://api.github.com/users/sunday-mayweather

#!/usr/bin/env python3
from random import random
import requests
import time
import json
import sys
import re

FORM = 1
FIELDS = 1
TITLE = 8
ID = 0
NAME = 1
DESCRIPTION = 2
TYPE = 3
VALUE = 4
OPTIONS = 1
URL = -2

types = {
    0: 'Short Answer',
    1: 'Paragraph',
    2: 'Radio',
    3: 'Dropdown',
    4: 'Checkboxes',
}

choice_types = ['Radio', 'Checkboxes', 'Dropdown']

def get_url(data):
    return 'https://docs.google.com/forms/d/' + data[URL] + '/formResponse'

def get_name(data):
    return data[FIELDS][TITLE]

def get_options(elem):
    options_raw = elem[VALUE][0][OPTIONS]
    return list(map(lambda l: l[0], options_raw))

def get_fields(data):
    fields = {}
    for elem in data[FORM][FIELDS]:
        field = {
            'description': elem[DESCRIPTION],
            'type': types.get(elem[TYPE]),
            'id': elem[VALUE][0][ID],
            'submit_id': 'entry.' + str(elem[VALUE][0][ID]),
        }

        if field['type'] in choice_types:
            field['options'] = get_options(elem)

        fields[elem[NAME]] = field
    return fields

def parse_data(data_str):
    data = json.loads(data_str)
    return {
        'url': get_url(data),
        'name': get_name(data),
        'fields': get_fields(data),
    }

def get_form(url):
    body = requests.get(url).text
    match = re.search(r'FB_PUBLIC_LOAD_DATA_ = ([^;]*);', body)
    if not match: return None
    data = parse_data(match.group(1))
    return data

def output(form):
    for name in form['fields']:
        field = form['fields'][name]
        print(name + ' (' + str(field['id']) + ')')
        if field['description']: print('> ' + field['description'])
        if 'options' in field:
            for option in field['options']:
                print('  - ' + option)
        print()

def submit(form):
    payload = {}
    for name in form['fields']:
        field = form['fields'][name]
        if field['type'] in choice_types and field['value'] not in field['options']:
            payload[field['submit_id']] = '__other_option__'
            payload[field['submit_id'] + '.other_option_response'] = field['value']
        else:
            payload[field['submit_id']] = field['value']

    return requests.post(form['url'], data=payload)


def main(url):
    headers = requests.utils.default_headers()
    headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/40.0.2214.85 Safari/537.36'
    })

    form = get_form(url)
    # output(form) # uncomment this to print out the contents of the form
    fields = form['fields']
    # fill out the fields here
    # fields['Field Name']['value'] = 'What you want to submit'
    fields['Greek Sing Show']['value'] = 'Booth > Greek Sing'
    fields['Best Ceremony']['value'] = 'Flag & Badge'
    fields['Mac & Cheese']['value'] = 'Lobster Mac'
    fields['What are your favorite colors?']['value'] = 'Purple, White, and Gold'
    fields['If a man is unsatisfied with himself, with who he is, and he wants to make of himself a better man, what must he do?']['value'] = \
        'Um... idk man, I thought it was Zach\'s job to tell me that?'

    num_submitions = 10 # change this to spam more/less
    for i in range(num_submitions):
        time.sleep(1 + random())
        submit(form)


if __name__ == '__main__':
    # put your url here
    url = 'https://goo.gl/forms/ymz5nFhAG4edk2jI2'
    main(url)
