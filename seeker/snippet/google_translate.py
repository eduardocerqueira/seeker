#date: 2021-08-31T02:19:02Z
#url: https://api.github.com/gists/37158ded08f312b0194fe2a32473ed65
#owner: https://api.github.com/users/RascalTwo

import requests
import json
import sys


header = {"User-Agent": "Chrome/41.0.2228.0"}

def translate(from_lang, to_lang, phrase):
    phrase = phrase.replace(" ", "%20")
    r = requests.get("https://translate.googleapis.com/translate_a/single?client=gtx&sl={}&tl={}&dt=t&q={}".format(from_lang, to_lang, phrase), headers=header)
    result = r.content.decode("utf-8").replace(",,", ", 0,").replace(",,", ", 0,")
    result = json.loads(result)
    print("From: {}\nTo  : {}".format(result[0][0][1], result[0][0][0]))

if len(sys.argv) != 4:
	print('Usage: python3 google_translate.py LANG_FROM LANG_TO PHRASE')
	sys.exit()

translate(*sys.argv[1:])

