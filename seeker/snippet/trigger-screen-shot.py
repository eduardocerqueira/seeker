#date: 2024-09-25T17:08:45Z
#url: https://api.github.com/gists/e2b81eb4c83f5d036ec3800497706885
#owner: https://api.github.com/users/zflat

#!/usr/bin/python3

# this used to live at https://gitlab.gnome.org/snippets/814
# but has since been deleted, the original author is unknown
# reuploading here for safe keeping

import dbus
import secrets
import re

from gi.repository import GLib
from dbus.mainloop.glib import DBusGMainLoop

class PortalBus:
    def __init__(self):
        DBusGMainLoop(set_as_default=True)

        self.bus = dbus.SessionBus()
        self.portal = self.bus.get_object('org.freedesktop.portal.Desktop', '/org/freedesktop/portal/desktop')

    def sender_name(self):
        return re.sub('\.', '_', self.bus.get_unique_name()).lstrip(':')

 "**********"  "**********"  "**********"  "**********"  "**********"d "**********"e "**********"f "**********"  "**********"r "**********"e "**********"q "**********"u "**********"e "**********"s "**********"t "**********"_ "**********"h "**********"a "**********"n "**********"d "**********"l "**********"e "**********"( "**********"s "**********"e "**********"l "**********"f "**********", "**********"  "**********"t "**********"o "**********"k "**********"e "**********"n "**********") "**********": "**********"
        return '/org/freedesktop/portal/desktop/request/%s/%s'%(self.sender_name(), token)

class PortalScreenshot:
    def __init__(self, portal_bus):
        self.portal_bus = portal_bus
        self.bus = portal_bus.bus
        self.portal = portal_bus.portal

    def request(self, callback, parent_window = ''):
        request_token = "**********"
        options = { 'handle_token': "**********"

        self.bus.add_signal_receiver(callback,
                                    'Response',
                                    'org.freedesktop.portal.Request',
                                    'org.freedesktop.portal.Desktop',
                                    self.portal_bus.request_handle(request_token))

        self.portal.Screenshot(parent_window, options, dbus_interface='org.freedesktop.portal.Screenshot')

    @staticmethod
 "**********"  "**********"  "**********"  "**********"  "**********"d "**********"e "**********"f "**********"  "**********"n "**********"e "**********"w "**********"_ "**********"u "**********"n "**********"i "**********"q "**********"u "**********"e "**********"_ "**********"t "**********"o "**********"k "**********"e "**********"n "**********"( "**********") "**********": "**********"
        return 'screen_shot_py_%s'%secrets.token_hex(16)

def callback(response, result):
    if response == 0:
        print(result['uri'])
    else:
        print("Failed to screenshot: %d"%response)

    loop.quit()

loop = GLib.MainLoop()
bus = PortalBus()
PortalScreenshot(bus).request(callback)

try:
    loop.run()
except KeyboardInterrupt:
    loop.quit()
