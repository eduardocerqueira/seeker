#date: 2022-10-26T16:57:01Z
#url: https://api.github.com/gists/4c0f98fbdccfa02ac8325863b24aa3e2
#owner: https://api.github.com/users/criticalth

import json

def export_json(dictionary, text_file):
    with open(text_file, "w") as outfile:
        json.dump(dictionary, outfile)


def import_json(text_file):

    with open(text_file, "r") as openfile:
        json_object = json.load(openfile)

    return json_object
  