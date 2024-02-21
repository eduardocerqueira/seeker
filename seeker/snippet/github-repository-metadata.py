#date: 2024-02-21T17:00:40Z
#url: https://api.github.com/gists/87166d9722d12315453a4e9cc5abcb1e
#owner: https://api.github.com/users/mvandermeulen

#! /usr/bin/env python3

####################################################################################################

import argparse
from datetime import datetime
import json
import os

# http://pygithub.readthedocs.io/en/latest/
from github import Github

####################################################################################################

parser = argparse.ArgumentParser(description='...')
parser.add_argument('--upload',
                    action='store_true', default=False,
                    help='upload from Github')
parser.add_argument('--json-path',
                    default='github-cache.json',
                    help='JSON cache file path')
parser.add_argument('--list',
                    action='store_true', default=False,
                    help='')

args = parser.parse_args()

####################################################################################################

class Repository:

    ##############################################

    def __init__(self, **kwargs):

        self._keys = kwargs.keys()
        for key, value in kwargs.items():
            setattr(self, key, value)

    ##############################################

    @staticmethod
    def _python_to_json(x):

        if isinstance(x, datetime):
            return str(x)
        else:
            return x

    ##############################################

    def to_json(self):

        return {key:self._python_to_json(getattr(self, key))
                for key in self._keys}

####################################################################################################

class Repositories:

    ##############################################

    def __init__(self):

        self._repositories = {}

    ##############################################

    def upload(self):

        token_path = "**********"
 "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"w "**********"i "**********"t "**********"h "**********"  "**********"o "**********"p "**********"e "**********"n "**********"( "**********"t "**********"o "**********"k "**********"e "**********"n "**********"_ "**********"p "**********"a "**********"t "**********"h "**********", "**********"  "**********"' "**********"r "**********"' "**********") "**********"  "**********"a "**********"s "**********"  "**********"f "**********": "**********"
            token = "**********"
            # token = "**********"

        print('Start upload')
        github = "**********"=token)

        for repository in github.get_user().get_repos():
            print('  {.name}'.format(repository))
            # http://pygithub.readthedocs.io/en/latest/github_objects/Repository.html
            # https://developer.github.com/v3/repos/
            keys = (
                #! 'topics',
                'created_at',
                'description',
                'fork',
                'forks_count',
                'full_name',
                'html_url',
                'language',
                'name',
                'network_count',
                'private',
                'pushed_at',
                'source',
                'stargazers_count',
                'subscribers_count',
                'updated_at',
                'watchers_count',
            )
            kwargs = {key:getattr(repository, key) for key in keys}
            source = kwargs['source']
            if source is not None:
                kwargs['source'] = source.full_name
            repository_data = Repository(**kwargs)
            self._repositories[repository_data.name] = repository_data
        print('Upload done')

    ##############################################

    def save(self, json_path):

        print('Write {}'.format(json_path))
        data = [repository.to_json() for repository in self._repositories.values()]
        with open(json_path, 'w') as fh:
            json.dump(data, fh, indent=4, sort_keys=True)

    ##############################################

    def load(self, json_path):

        print('Load {}'.format(json_path))
        with open(json_path, 'r') as fh:
            data = json.load(fh)
        for repository_data in data:
            repository_data = Repository(**repository_data)
            self._repositories[repository_data.name] = repository_data

    ##############################################

    @property
    def names(self):
        return sorted(self._repositories.keys())

    ##############################################

    def __iter__(self):

        # return iter(self._repositories.values())
        for name in self.names:
            repository = self._repositories[name]
            yield repository

    ##############################################

    def __getitem__(self, name):
        return self._repositories[name]

    ##############################################

    @property
    def forks(self):

        for repository in self:
            if repository.fork:
                yield repository

    ##############################################

    @property
    def fork_names(self):

        for repository in self.forks:
            yield repository.name

    ##############################################

    def by_star(self, names):

        repositories = [self._repositories[name] for name in sorted(names)]
        get_key = lambda x: x.stargazers_count # '{:3}{}'.format(x.stargazers_count, x.name)
        return sorted(repositories, key=get_key, reverse=True)

####################################################################################################

repositories = Repositories()
if args.upload:
    repositories.upload()
    repositories.save(args.json_path)
else:
    repositories.load(args.json_path)

if args.list:
    for repository in repositories:
        fork = 'F' if repository.fork else ' '
        print("{} '{}',".format(fork, repository.name))
