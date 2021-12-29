#date: 2021-12-29T17:17:02Z
#url: https://api.github.com/gists/f85e3319c7a9a5d2e05e2b0d701b215b
#owner: https://api.github.com/users/marcosricardoss

from django.core import serializers
from myproject.myapp import models
data = serializers.serialize("json", models.MyModel.objects.all())
out = open("dump.json", "w")
out.write(data)
out.close()