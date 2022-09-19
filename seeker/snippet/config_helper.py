#date: 2022-09-19T17:22:23Z
#url: https://api.github.com/gists/08e002dd0e5b5bd5daa462d8a490e6be
#owner: https://api.github.com/users/GGontijo

import json

class Config:
    '''Singleton approach'''

    _instance = None

    def __init__(self) -> None:
        CONFIG_FILE = 'conf.json'
        with open(CONFIG_FILE, 'r') as config:
            self.__config = json.load(config)
        
    def get_config(self, var: str) -> str:
        value = self.__config[var]
        return value

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance