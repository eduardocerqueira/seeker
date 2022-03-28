#date: 2022-03-28T17:14:31Z
#url: https://api.github.com/gists/bf3e0d02cc2fe02ba5ff69df54846762
#owner: https://api.github.com/users/agiletalk

import os
import sys
import yaml
import json

def get_file_list_yaml(path):
	return [file for file in os.listdir(path) if file.endswith(".yaml")]

def yaml_to_json(yaml_file, json_file):
	with open(yaml_file, 'r') as yaml_in, open(json_file, 'w') as json_out:
		yaml_object = yaml.safe_load(yaml_in)
		json.dump(yaml_object, json_out)

if __name__ == '__main__':
	file_list_yaml = get_file_list_yaml(sys.argv[1])
	for file in file_list_yaml:
		file_name, ext = os.path.splitext(file)
		yaml_to_json(file, file_name+".json")