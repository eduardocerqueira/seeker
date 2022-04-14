#date: 2022-04-14T17:06:44Z
#url: https://api.github.com/gists/0551ef9ab6f8f3c5c5c6b0b997efae7e
#owner: https://api.github.com/users/davidejones

import argparse
from pprint import pprint

import requests
import pathlib

session = requests.Session()


def get_resources(options):
    url = f'https://rest.api.transifex.com/resources?filter[project]=o:{options.org}:p:{options.project}'
    items = session.get(url, headers={'Authorization': f'Bearer {options.api_key}'}).json()
    while items:
        for item in items['data']:
            yield item
        if items['links'] and items['links']['next']:
            items = session.get(items['links']['next'], headers={'Authorization': f'Bearer {options.api_key}'}).json()
        else:
            items = None


def main(options):
    missing_files = []
    word_counts = []
    for resource in get_resources(options):
        attrs = resource.get('attributes', {})
        path = ''.join(attrs.get('categories', ['']))
        if path.startswith('content/'):
            path = path.replace('content/', 'content/en/')
        path_to_file = pathlib.Path(f"{options.local}{path}/{attrs.get('name')}")
        if not path_to_file.exists():
            missing_files.append(str(path_to_file))
    pprint(missing_files)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Finds transifex files to delete')
    parser.add_argument('org', help='transifex org name')
    parser.add_argument('project', help='transifex project name')
    parser.add_argument('api_key', help='transifex apikey')
    parser.add_argument('local', help='path to local site')
    options = parser.parse_args()
    main(options)
