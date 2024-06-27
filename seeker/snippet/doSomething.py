#date: 2024-06-27T17:04:03Z
#url: https://api.github.com/gists/1ef1631291c8363b9c142e8bf551f293
#owner: https://api.github.com/users/corei8

import os

def article_data() -> dict:
	"""
	json file keeps the record of the "age" of the newest file in the directory.
	"""
	with open('./last_ocr_file.json', 'r') as database:
		return json.load(database)

def check_ocred(file: str, directory: str) -> bool:
	"""
	Check each file for age. 
	"""
	data = article_data()
	html, md = filename_html(file), filename_md(file)
	path_md = MKDN_PATH+directory+'/'+md
	path_html = HTML_PATH+directory+'/'+html
	if not os.path.isfile(path_html):
		if not os.path.isfile(path_md):
			# TODO adjust this so that page is fetched form here.
			return 404
		else:
			build_article(file=file, directory=directory)
	else:
		data = article_data()
		info = os.stat(path_html)
		try:
			if info.st_mtime == data[directory][file]['mod']:
				return False
			else:
				return True
		except KeyError:
			build_article(file=file, directory=directory)
			return False