#date: 2022-03-11T17:07:33Z
#url: https://api.github.com/gists/60d766b8d6cd970b42eaf967b8dac3ff
#owner: https://api.github.com/users/SphinxKnight

import os
import hashlib

ref_locale = 'en-us'
ref_path = 'content/files/' + ref_locale + '/'

locale = 'fr'
locale_path = 'translated-content/files/' + locale + '/'

dict_files={}
for r, d, f in os.walk(ref_path):
    for file in f :
        if not('.md' in file) and not('.html' in file):
            full_path = os.path.join(r, file)
            file_b = open(full_path,"rb")
            content = file_b.read()
            file_slug= full_path.split(ref_locale)[1]
            dict_files[file_slug] = hashlib.sha256(content).hexdigest()

# print(dict_files)
spared_size = 0
for r, d, f in os.walk(locale_path):
    for file in f :
        if not('.md' in file) and not('.html' in file):
            full_path = os.path.join(r, file)
            file_slug= full_path.split(locale)[1]
            if file_slug in dict_files:
                file_b = open(full_path,"rb")
                content = file_b.read()
                locale_file_hash = hashlib.sha256(content).hexdigest()
                if locale_file_hash == dict_files[file_slug]:
                    spared_size = spared_size + os.path.getsize(full_path)
                    print(full_path)
print(spared_size)