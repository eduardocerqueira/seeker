#date: 2022-02-21T16:54:05Z
#url: https://api.github.com/gists/e143804bf4b570a44e3c1db6b1deba23
#owner: https://api.github.com/users/kamakazix

import os

from gi.repository import Nautilus, GObject

class ColumnExtension(GObject.GObject, Nautilus.MenuProvider):
    def __init__(self):
        pass
    def menu_activate_cb(self, menu, file):
        # Command to run terminal
        os.system("terminator --working-directory=" + file.get_location().get_path())

    def create_menu_item(self):
        return Nautilus.MenuItem(name='TerminatorExtension::Open_Terminator',
                                     label='Open in Terminator',
                                     tip='',
                                     icon='')

    def get_background_items(self, window, file):
        item = self.create_menu_item()
        item.connect('activate', self.menu_activate_cb, file)
        return item,

    def get_file_items(self, menu, files):
        if len(files) != 1:
            return
        file = files[0]
        if not file.is_directory():
            return
        item = self.create_menu_item()
        item.connect('activate', self.menu_activate_cb, file)
        return item,
