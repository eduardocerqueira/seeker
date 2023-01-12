#date: 2023-01-12T17:01:23Z
#url: https://api.github.com/gists/bc7d6fa22ecd9c00dbfb6520c0824caf
#owner: https://api.github.com/users/TimmyT123

import base64
import time
import urllib2

import cv2
import numpy as np


"""
Examples of objects for image frame aquisition from both IP and
physically connected cameras

Requires:
 - opencv (cv2 bindings)
 - numpy
"""


class ipCamera(object):

 "**********"  "**********"  "**********"  "**********"  "**********"d "**********"e "**********"f "**********"  "**********"_ "**********"_ "**********"i "**********"n "**********"i "**********"t "**********"_ "**********"_ "**********"( "**********"s "**********"e "**********"l "**********"f "**********", "**********"  "**********"u "**********"r "**********"l "**********", "**********"  "**********"u "**********"s "**********"e "**********"r "**********"= "**********"N "**********"o "**********"n "**********"e "**********", "**********"  "**********"p "**********"a "**********"s "**********"s "**********"w "**********"o "**********"r "**********"d "**********"= "**********"N "**********"o "**********"n "**********"e "**********") "**********": "**********"
        self.url = url
        auth_encoded = base64.encodestring('%s: "**********":-1]

        self.req = urllib2.Request(self.url)
        self.req.add_header('Authorization', 'Basic %s' % auth_encoded)

    def get_frame(self):
        response = urllib2.urlopen(self.req)
        img_array = np.asarray(bytearray(response.read()), dtype=np.uint8)
        frame = cv2.imdecode(img_array, 1)
        return frame


class Camera(object):

    def __init__(self, camera=0):
        self.cam = cv2.VideoCapture(camera)
        if not self.cam:
            raise Exception("Camera not accessible")

        self.shape = self.get_frame().shape

    def get_frame(self):
        _, frame = self.cam.read()
        return frame
