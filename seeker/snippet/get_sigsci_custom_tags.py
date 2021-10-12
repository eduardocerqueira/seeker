#date: 2021-10-12T17:01:50Z
#url: https://api.github.com/gists/2a2b4b75e70ab45cdee3e53cb2303075
#owner: https://api.github.com/users/BrooksCunningham

from SigSci import *
import os
import json


# setup sigsci api module
sigsci = SigSciAPI()

# required variables
sigsci.email = os.environ['SIGSCI_EMAIL']
sigsci.api_token = os.environ['SIGSCI_API_TOKEN']
sigsci.corp  = os.environ['SIGSCICORP']
sigsci.site  = os.environ['SIGSCI_SITE']

print(sigsci.get_custom_tags())
