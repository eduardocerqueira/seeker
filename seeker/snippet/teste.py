#date: 2024-01-05T16:59:16Z
#url: https://api.github.com/gists/ec29eaf2c8b6bf37be25a61135b7a3f4
#owner: https://api.github.com/users/thiagopilz

#!/usr/bin/env python

import os, sys
import filecmp
import re
from distutils import dir_util
import shutil

holderlist = []

def compareme(dir1, dir2):
    dircomp = filecmp.dircmp(dir1, dir2)
    only_in_one = dircomp.left_only
    diff_in_one = dircomp.diff_files
    dirpath = os.path.abspath(dir1)
    [holderlist.append(os.path.abspath(os.path.join(dir1, x))) for x in only_in_one]
    [holderlist.append(os.path.abspath(os.path.join(dir1, x))) for x in diff_in_one]
    if len(dircomp.common_dirs) > 0:
        for item in dircomp.common_dirs:
            compareme(
                os.path.abspath(os.path.join(dir1, item)),
                os.path.abspath(os.path.join(dir2, item)),
)
return holderlist

def main():
if len(sys.argv) > 3:
    dir1 = sys.argv[1]
    dir2 = sys.argv[2]
    dir3 = sys.argv[3]
else:
print "Usage: ", sys.argv[0], "currentdir olddir difference"
sys.exit(1)

if not dir3.endswith("/"):
dir3 = dir3 + "/"

source_files = compareme(dir1, dir2)
dir1 = os.path.abspath(dir1)
dir3 = os.path.abspath(dir3)
destination_files = []
new_dirs_create = []
for item in source_files:
    destination_files.append(re.sub(dir1, dir3, item))
for item in destination_files:
    new_dirs_create.append(os.path.split(item)[0])
for mydir in set(new_dirs_create):
    if not os.path.exists(mydir):
        os.makedirs(mydir)
# copy pair
copy_pair = zip(source_files, destination_files)
for item in copy_pair:
        if os.path.isfile(item[0]):
            shutil.copyfile(item[0], item[1])

if __name__ == "__main__":
    main()
#--------


#-Geração da pasta com arquivos atualizados apenas!!
#Compara a diferença das duas pastas e copía os arquivos para uma terceira.
#EX:
# python teste.py /usr9/AtuDatacenter/web/CompusisWebMS-2305/ /usr/local/tomcat9/webapps/premiereweb/ /usr9/AtuDatacenter/web/unzip/

#-Copy Modified Files with Rsync
# rsync -ur /usr9/AtuDatacenter/web/unzip/ /usr/local/tomcat9/webapps/premiereweb/