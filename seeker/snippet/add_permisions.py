#date: 2022-02-21T17:09:25Z
#url: https://api.github.com/gists/10b55ca897408972d45e5d0313dc41dd
#owner: https://api.github.com/users/xtornasol512

"""  Little snippet for looking for permissions and add them to a group """

from django.contrib.auth.models import Group
from django.contrib.auth.models import Permission

from <your-model-folder>.models import <MODEL>, <MODEL-N>

gp = Group.objects.get(name="<GROUP_NAME>")
qs = Permission.objects.filter(codename__icontains="<FOLDER-NAME>")

for p in qs:
  gp.permissions.add(p)

  
