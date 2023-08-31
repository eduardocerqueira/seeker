#date: 2023-08-31T17:04:43Z
#url: https://api.github.com/gists/ac1e5fc2b875d56e875e97ca9aebd582
#owner: https://api.github.com/users/joshuaboud

#!/usr/bin/env python3

"""
Watch org.kde.KWin/ColorCorrect.PropertiesChanged signal and set OpenRGB profile accordingly

Author: Joshua Boudreau 2023

# How to

1. Save nightmode_rgb.py anywhere (e.g. `/usr/local/bin`)
2. Edit `PROFILES` list
3. `sudo chmod +x /usr/local/bin/nightmode_rgb.py`
4. KDE settings -> Startup and Shutdown -> Autostart -> Add... -> Add Application -> Browse -> select nightmode_rgb.py

"""

import dbus
from openrgb import OpenRGBClient
from dbus.mainloop.glib import DBusGMainLoop
from gi.repository import GLib
from typing import List, Tuple

PROFILE_LOCAL = True
PROFILE_SYS = False

"""
List of color temperature thresholds and OpenRGB profile names

[(THRES, PROFILE, LOCAL), ...]
THRES: int - color temperature threshold (uses profile if temp above threshold)
PROFILE: str - name of OpenRGB profile
LOCAL: bool - local user OpenRGB profile if True, system profile if false
"""
PROFILES: List[Tuple[int, str, bool]] = [
  (0, 'cyber_warm', PROFILE_LOCAL),
  (5000, 'cyber', PROFILE_LOCAL),
]

RGB = OpenRGBClient(protocol_version=3)
DBusGMainLoop(set_as_default=True)
CURRENT_PROFILE: str = None


def pick_profile(color_temp: int) -> Tuple[str, bool]:
  for thres, profile, local in reversed(PROFILES):
    if color_temp > thres:
      return profile, local
  return PROFILES[0]


def update_profile(color_temp: int) -> None:
  global CURRENT_PROFILE
  profile, local = pick_profile(color_temp)
  if profile == CURRENT_PROFILE:
    return
  print(f"setting OpenRGB profile '{profile}'")
  RGB.load_profile(profile, local=local)
  CURRENT_PROFILE = profile


def on_properties_changed(sender: str, properties: dbus.Dictionary, args: dbus.Array):
  if 'currentTemperature' in properties:
    update_profile(properties['currentTemperature'])


def main():
  bus = dbus.SessionBus()
  color_correct_proxy = bus.get_object('org.kde.KWin', '/ColorCorrect')
  color_correct_iface = dbus.Interface(color_correct_proxy, 'org.freedesktop.DBus.Properties')
  color_correct_iface.connect_to_signal('PropertiesChanged', on_properties_changed)

  color_temp = color_correct_iface.Get('org.kde.kwin.ColorCorrect', 'currentTemperature')
  update_profile(color_temp)

  loop = GLib.MainLoop()
  loop.run()


if __name__ == '__main__':
  main()
