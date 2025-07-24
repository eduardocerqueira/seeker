#date: 2025-07-24T17:05:43Z
#url: https://api.github.com/gists/16d70f0cfdb3131ce87275ac8c4b7216
#owner: https://api.github.com/users/lcrs

# Downloads URLs from a Pocket export CSV into one folder per month
# Creates a .webloc file per URL to open in a browser and also tries to download the page
# Pass path to .csv file like:
# 	python3 pocketdownload.py pocket_export_20250526.csv

import sys, os, datetime, time, subprocess

csv = open(sys.argv[1]).readlines()
csv.pop(0)
csv.sort(key=lambda x: x.split(',')[-3])

# First just make a .webloc file from each URL
for p in csv:
	url, when, tags, status = p.split(',')[-4:]
	yyyymm = datetime.datetime.fromtimestamp(int(when)).strftime('%Y%m')
	escapedurl = url.replace('/', '_').replace(':', '_')

	print(f'Creating .webloc {yyyymm}/weblocs/{escapedurl}.webloc')

	try:
		os.makedirs(f'downloads/{yyyymm}/weblocs')
	except:
		pass

	xmlurl = url.replace('&', '&amp;')
	webloc = 	('<?xml version="1.0" encoding="UTF-8"?>\n'
				'<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">\n'
				'<plist version="1.0">\n'
				'<dict>\n'
				'        <key>URL</key>\n'
				f'        <string>{xmlurl}</string>\n'
				'</dict>\n'
				'</plist>\n')

	with open(f'downloads/{yyyymm}/weblocs/{escapedurl}.webloc', 'w') as w:
		w.write(webloc)

# Then download the original page for each URL
for p in csv:
	url, when, tags, status = p.split(',')[-4:]
	yyyymm = datetime.datetime.fromtimestamp(int(when)).strftime('%Y%m')
	escapedurl = url.replace('/', '_').replace(':', '_')
	downloaddir = f'downloads/{yyyymm}/{escapedurl}'

	print(f'Downloading {yyyymm}/{escapedurl}')

	try:
		os.makedirs(downloaddir)
	except:
		pass

	if('twitter.com' in url or 'x.com' in url):
		url = url.replace('twitter.com', 'nitter.privacyredirect.com')
		url = url.replace('x.com', 'nitter.privacyredirect.com')
	subprocess.run(('wget', '-e', 'robots=off', '-P', downloaddir, '-E', '-H', '-k', '-p', url))

	time.sleep(2)
