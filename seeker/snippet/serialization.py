#date: 2022-06-02T17:00:38Z
#url: https://api.github.com/gists/b6c685d4938012e995219e5f374c3340
#owner: https://api.github.com/users/matthewjberger

import dill
import json

jsonStr =  '{ "name":"John", "age":30, "city":"New York"}'

def save():
	print("save")
	print(jsonStr)
	print(json.loads(jsonStr))
	print(dill.dumps(json.loads(jsonStr)))

	return dill.dumps(json.loads(jsonStr))

def load(b):
	print("load")
	print(b)
	print(dill.loads(b))
	print(json.dumps(dill.loads(b)))

binary = save()
load(binary)