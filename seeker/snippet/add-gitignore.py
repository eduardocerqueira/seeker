#date: 2022-12-07T17:04:59Z
#url: https://api.github.com/gists/9bf2773c361f3c0883185cbec687356e
#owner: https://api.github.com/users/paschembri

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import os
import urllib.request

gitignore_url = {
    'javascript': 'https://raw.githubusercontent.com/github/gitignore/main/Node.gitignore',
    'python': 'https://raw.githubusercontent.com/github/gitignore/main/Python.gitignore',
}


def main():
    parser = argparse.ArgumentParser(
        description='''Create a .gitignore file in the current directory.

        The script retrieves gitignore templates from github.
        ''',
        usage='add-gitignore --lang=javascript',
    )
    parser.add_argument(
        '--lang',
        action='store',
        default='python',
        help='Ex.: javascript',
    )

    args = parser.parse_args()

    try:
        url = gitignore_url[args.lang]
    except KeyError:
        print(f'{args.lang} is not supported yet.')

    urllib.request.urlretrieve(url, ".gitignore")
    print(f'Downloaded {url} > .gitignore')


if __name__ == '__main__':
    main()
