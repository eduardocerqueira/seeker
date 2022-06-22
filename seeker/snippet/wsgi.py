#date: 2022-06-22T17:21:53Z
#url: https://api.github.com/gists/441f6630a7b10f201a88d7903eeeb938
#owner: https://api.github.com/users/AnvilFox1965

import sys
import os
from django.core.handlers.wsgi import WSGIHandler

# If using virtualenvs
import site
envpath = "/Users/jero/dev/thm_env/lib/python2.7/site-packages"
site.addsitedir(envpath)

sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), 'THM'))

os.environ['DJANGO_SETTINGS_MODULE'] = 'settings'
application = WSGIHandler()
