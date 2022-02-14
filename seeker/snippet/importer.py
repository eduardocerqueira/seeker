#date: 2022-02-14T17:04:27Z
#url: https://api.github.com/gists/12aaf4d2e45d2499ae3ce672f0ef81e9
#owner: https://api.github.com/users/kamuridesu

import os

def importer():
    modules = ["man", "test"]  # list with all modules name
    files = [x for x in os.listdir() if x.endswith(".py") and x != "__init__.py" and x != __file__]  # get all .py files
    order = {}  # dict to hold our data
    for file in files:
        with open(file, "r", encoding="utf-8") as f:
            text = f.readline().strip("\n")
            if text.startswith("#"):  # if indexed
                order[modules[int(text[1:]) - 1]] = file  # add file on dict
    globals().update({x: __import__(order[x][:-3]) for x in order})  # add imports to globals

importer()