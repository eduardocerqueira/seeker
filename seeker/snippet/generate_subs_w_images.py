#date: 2022-07-20T17:09:26Z
#url: https://api.github.com/gists/4349fb550218a6492b063ae3510db091
#owner: https://api.github.com/users/lognaturel

#!/usr/bin/env python
# -*- coding: utf-8 -*-

import uuid
import random
import os
import string
from PIL import Image
import numpy

sub_template = "<?xml version='1.0' ?><data id=\"media_stress_test_1\" instanceID=\"uuid:%s\" version=\"1\" xmlns:xsd=\"http://www.w3.org/2001/XMLSchema\" xmlns:h=\"http://www.w3.org/1999/xhtml\" xmlns:jr=\"http://openrosa.org/javarosa\"><meta><instanceID>uuid:%s</instanceID></meta><text>%s</text><integer>%s</integer><image>%s.png</image></data>"


for i in range(0, 219):
	uid = uuid.uuid4()
	text = ''.join(random.choices(string.ascii_letters + string.digits, k=16))
	integer = random.randint(0, 10000)
	filename = ''.join(random.choices(string.ascii_letters + string.digits, k=16))

	instance = sub_template % (uid, uid, text, integer, filename)

	instance_folder = os.path.join("instances/", str(uid))
	os.mkdir(instance_folder)
	f = open(os.path.join(instance_folder, "submission.xml"), "w+")
	f.write(instance)
	f.close()

	imarray = numpy.random.rand(640, 640, 3) * 255
	im = Image.fromarray(imarray.astype('uint8')).convert('RGBA')
	im.save(str(instance_folder) + "/" + filename + ".png")