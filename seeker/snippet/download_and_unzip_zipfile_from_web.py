#date: 2023-01-13T16:54:31Z
#url: https://api.github.com/gists/4d639979143a705b8ffd51b3d0d31da0
#owner: https://api.github.com/users/kbn-gh

import requests
import io
import zipfile
import os.path as op

# Weblink wrong replace with correct one
WEB_LINK = 'https://biolinuxtest.wustl.edu/data.zip'
PATH_DOWNLOAD = '/hlabhome/kiranbn/tmp_app/'

# check folder exist
if not op.exists(PATH_DOWNLOAD):
    raise FileNotFoundError("Folder {} not found".format(PATH_DOWNLOAD))

# Download and unzip to PATH_DOWNLOAD
response = requests.get(WEB_LINK)
with zipfile.ZipFile(io.BytesIO(response.content), 'r') as zip_f:
    zip_f.extractall(PATH_DOWNLOAD)